import express from "express";
import makeWASocket, {
  DisconnectReason,
  fetchLatestBaileysVersion,
  jidNormalizedUser,
  useMultiFileAuthState,
} from "@whiskeysockets/baileys";
import qrcode from "qrcode-terminal";
import fs from "fs/promises";
import path from "path";

const port = Number.parseInt(process.env.ARKA_WHATSAPP_BRIDGE_PORT ?? "8091", 10);
const controlplaneBaseUrl =
  process.env.ARKA_WHATSAPP_CONTROLPLANE_BASE_URL ?? "http://127.0.0.1:8080";
const accountId = process.env.ARKA_WHATSAPP_ACCOUNT_ID ?? "default";
const authDir =
  process.env.ARKA_WHATSAPP_AUTH_DIR ??
  path.join(process.cwd(), "data", "whatsapp-web", "auth");

const state = {
  sock: null,
  connectionState: "needs_login",
  loginSessionId: null,
  qrTerminal: null,
  lastError: null,
  reconnectTimer: null,
  reconnectAttempts: 0,
  hasConnected: false,
};

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/status", (_req, res) => {
  res.json(currentStatus());
});

app.post("/login/start", async (_req, res) => {
  try {
    await ensureSocket(true);
    const started = await waitForLoginSignal();
    res.json(started);
  } catch (error) {
    res.status(500).json({ error: String(error) });
  }
});

app.post("/login/complete", async (req, res) => {
  if (req.body?.login_session_id && req.body.login_session_id !== state.loginSessionId) {
    return res.status(400).json({ error: "login session does not match active QR session" });
  }
  try {
    for (let attempt = 0; attempt < 60; attempt += 1) {
      if (state.connectionState === "ready") {
        return res.json(currentStatus());
      }
      await sleep(1000);
    }
    return res.status(409).json({ error: "whatsapp web session is not ready yet" });
  } catch (error) {
    res.status(500).json({ error: String(error) });
  }
});

app.post("/logout", async (_req, res) => {
  try {
    if (state.sock) {
      await state.sock.logout();
    }
    state.sock = null;
    state.connectionState = "needs_login";
    state.loginSessionId = null;
    state.qrTerminal = null;
    state.lastError = null;
    await fs.rm(authDir, { recursive: true, force: true });
    res.json(currentStatus());
  } catch (error) {
    state.lastError = String(error);
    res.status(500).json({ error: state.lastError });
  }
});

app.post("/messages", async (req, res) => {
  const to = resolveRecipientJid(req.body?.to);
  const kind = typeof req.body?.kind === "string" ? req.body.kind.trim() : "text";
  if (!to) {
    return res.status(400).json({ error: "to is required" });
  }
  try {
    await ensureSocket(false);
    if (state.connectionState !== "ready" || !state.sock) {
      return res.status(409).json({ error: "whatsapp web is not connected" });
    }
    const payload = outboundMessagePayload(req.body, kind);
    console.log(
      `Sending WhatsApp ${kind} reply to ${to}: ${truncateForLog(describeOutboundPayload(payload))}`
    );
    await state.sock.sendMessage(jidNormalizedUser(to), payload);
    res.json({ ok: true });
  } catch (error) {
    state.lastError = String(error);
    res.status(500).json({ error: state.lastError });
  }
});

await fs.mkdir(authDir, { recursive: true });

app.listen(port, () => {
  console.log(`WhatsApp Web bridge listening on http://127.0.0.1:${port}`);
  console.log(`WhatsApp auth directory: ${authDir}`);
  bootstrapSocket();
});

async function ensureSocket(allowQr) {
  if (state.sock && state.connectionState === "ready") {
    return state.sock;
  }
  if (state.sock && !allowQr) {
    return state.sock;
  }

  await fs.mkdir(authDir, { recursive: true });
  const { state: authState, saveCreds } = await useMultiFileAuthState(authDir);
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({
    auth: authState,
    version,
    printQRInTerminal: false,
    browser: ["Arka", "Chrome", "1.0.0"],
    syncFullHistory: false,
  });

  sock.ev.on("creds.update", saveCreds);
  sock.ev.on("connection.update", async (update) => {
    if (update.qr) {
      state.connectionState = "needs_login";
      state.loginSessionId = state.loginSessionId ?? crypto.randomUUID();
      state.qrTerminal = renderQr(update.qr);
      state.lastError = null;
      process.stdout.write(
        `\n=== WhatsApp QR (${state.loginSessionId}) ===\n${state.qrTerminal}\n`
      );
    }

    if (update.connection === "open") {
      state.connectionState = "ready";
      state.qrTerminal = null;
      state.loginSessionId = null;
      state.lastError = null;
      state.reconnectAttempts = 0;
      state.hasConnected = true;
      console.log("WhatsApp Web connected");
    }

    if (update.connection === "close") {
      const code = update.lastDisconnect?.error?.output?.statusCode;
      const loggedOut = code === DisconnectReason.loggedOut;
      const shouldRestart =
        code === DisconnectReason.restartRequired ||
        code === DisconnectReason.connectionClosed ||
        code === DisconnectReason.timedOut ||
        code === DisconnectReason.unavailableService;
      state.sock = null;
      state.connectionState = "needs_login";
      state.qrTerminal = null;
      state.lastError = loggedOut
        ? "session logged out"
        : shouldRestart
          ? `connection closed; restarting (${code})`
          : `connection closed (${code ?? "unknown"})`;
      console.log(
        `WhatsApp Web connection closed: code=${code ?? "unknown"} loggedOut=${loggedOut} restart=${shouldRestart}`
      );
      if (loggedOut) {
        state.loginSessionId = null;
        state.reconnectAttempts = 0;
        state.hasConnected = false;
        await fs.rm(authDir, { recursive: true, force: true });
      } else if (shouldRestart) {
        state.reconnectAttempts += 1;
        console.log(`WhatsApp Web connection closed with ${code}; restarting socket`);
        scheduleReconnect();
      } else if (!state.hasConnected) {
        state.reconnectAttempts += 1;
        console.log("WhatsApp Web failed before reaching ready; retrying bootstrap");
        scheduleReconnect();
      }
    }
  });

  sock.ev.on("messages.upsert", async (event) => {
    console.log(
      `WhatsApp upsert: type=${event.type} messages=${event.messages?.length ?? 0}`
    );
    if (event.type !== "notify") {
      return;
    }
    for (const message of event.messages) {
      if (!message.message || message.key.fromMe) {
        continue;
      }
      const remoteJid = message.key.remoteJid ?? "";
      if (remoteJid.endsWith("@g.us")) {
        continue;
      }
      const normalized = unwrapMessageContent(message.message);
      const text = extractText(normalized);
      if (!text.trim()) {
        console.log(
          `Ignoring inbound WhatsApp message without text: remoteJid=${remoteJid} keys=${Object.keys(normalized).join(",")}`
        );
        continue;
      }
      console.log(
        `Forwarding inbound WhatsApp text from ${remoteJid}: ${truncateForLog(text)}`
      );
      await forwardInbound({
        message_id: message.key.id,
        account_id: accountId,
        conversation_id: remoteJid,
        from_user_id: remoteJid,
        text,
        quoted_message_id: normalized.extendedTextMessage?.contextInfo?.stanzaId ?? null,
        quoted_text: null,
      });
    }
  });

  state.sock = sock;
  return sock;
}

async function waitForLoginSignal() {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (state.connectionState === "ready") {
      return {
        account_id: accountId,
        login_session_id: state.loginSessionId ?? crypto.randomUUID(),
        qr_code: "already connected",
      };
    }
    if (state.qrTerminal && state.loginSessionId) {
      return {
        account_id: accountId,
        login_session_id: state.loginSessionId,
        qr_code: state.qrTerminal,
      };
    }
    await sleep(500);
  }
  throw new Error("timed out waiting for whatsapp qr");
}

async function forwardInbound(payload) {
  console.log(
    `POST ${controlplaneBaseUrl.replace(/\/$/, "")}/channels/whatsapp/inbound message_id=${payload.message_id} from=${payload.from_user_id}`
  );
  const response = await fetch(`${controlplaneBaseUrl.replace(/\/$/, "")}/channels/whatsapp/inbound`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.text();
    console.warn(`Failed to forward inbound WhatsApp message: ${response.status} ${body}`);
    return;
  }
  const body = await response.text();
  console.log(`Inbound WhatsApp message accepted by Arka: ${body}`);
}

function currentStatus() {
  return {
    account_id: accountId,
    connection_state: state.connectionState,
    active_login_session_id: state.loginSessionId,
    last_error: state.lastError,
  };
}

function renderQr(rawQr) {
  let rendered = "";
  qrcode.generate(rawQr, { small: true }, (output) => {
    rendered = output;
  });
  return rendered || rawQr;
}

function normalizeRecipient(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value.replace(/@s\.whatsapp\.net$/, "").replace(/[^\d]/g, "");
}

function resolveRecipientJid(value) {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.includes("@")) {
    return trimmed;
  }
  const digits = normalizeRecipient(trimmed);
  return digits ? `${digits}@s.whatsapp.net` : "";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function unwrapMessageContent(message) {
  let current = message;
  while (current) {
    if (current.ephemeralMessage?.message) {
      current = current.ephemeralMessage.message;
      continue;
    }
    if (current.viewOnceMessage?.message) {
      current = current.viewOnceMessage.message;
      continue;
    }
    if (current.viewOnceMessageV2?.message) {
      current = current.viewOnceMessageV2.message;
      continue;
    }
    if (current.documentWithCaptionMessage?.message) {
      current = current.documentWithCaptionMessage.message;
      continue;
    }
    if (current.editedMessage?.message) {
      current = current.editedMessage.message;
      continue;
    }
    break;
  }
  return current ?? {};
}

function extractText(message) {
  return (
    message.conversation ??
    message.extendedTextMessage?.text ??
    message.imageMessage?.caption ??
    message.videoMessage?.caption ??
    ""
  );
}

function truncateForLog(text) {
  return text.length > 120 ? `${text.slice(0, 117)}...` : text;
}

function outboundMessagePayload(body, kind) {
  if (kind === "text") {
    const text = typeof body?.text === "string" ? body.text.trim() : "";
    if (!text) {
      throw new Error("text is required for text messages");
    }
    return { text };
  }

  if (kind === "image") {
    const url = typeof body?.url === "string" ? body.url.trim() : "";
    if (!url) {
      throw new Error("url is required for image messages");
    }
    const caption = typeof body?.caption === "string" ? body.caption.trim() : "";
    return {
      image: { url },
      ...(caption ? { caption } : {}),
    };
  }

  if (kind === "document") {
    const url = typeof body?.url === "string" ? body.url.trim() : "";
    const fileName = typeof body?.file_name === "string" ? body.file_name.trim() : "";
    if (!url || !fileName) {
      throw new Error("url and file_name are required for document messages");
    }
    const caption = typeof body?.caption === "string" ? body.caption.trim() : "";
    const mimeType = typeof body?.mime_type === "string" ? body.mime_type.trim() : "";
    return {
      document: { url },
      fileName,
      ...(mimeType ? { mimetype: mimeType } : {}),
      ...(caption ? { caption } : {}),
    };
  }

  throw new Error(`unsupported outbound message kind: ${kind}`);
}

function describeOutboundPayload(payload) {
  if (typeof payload.text === "string" && payload.text) {
    return payload.text;
  }
  if (typeof payload.caption === "string" && payload.caption) {
    return payload.caption;
  }
  if (typeof payload.fileName === "string" && payload.fileName) {
    return payload.fileName;
  }
  if (payload.image?.url) {
    return payload.image.url;
  }
  if (payload.document?.url) {
    return payload.document.url;
  }
  return JSON.stringify(payload);
}

async function bootstrapSocket() {
  try {
    if (!state.hasConnected && state.reconnectAttempts >= 2) {
      console.log("Resetting local WhatsApp auth state after repeated pre-open failures");
      await fs.rm(authDir, { recursive: true, force: true });
      await fs.mkdir(authDir, { recursive: true });
      state.loginSessionId = null;
      state.qrTerminal = null;
      state.reconnectAttempts = 0;
    }
    await ensureSocket(true);
  } catch (error) {
    state.lastError = String(error);
    console.error(`Failed to bootstrap WhatsApp Web socket: ${state.lastError}`);
    state.reconnectAttempts += 1;
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (state.reconnectTimer) {
    return;
  }
  state.reconnectTimer = setTimeout(async () => {
    state.reconnectTimer = null;
    await bootstrapSocket();
  }, 1000);
}
