//! Turn-scoped todo parsing, validation, and mutation helpers.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MANDATORY_TODO_GENERATE_HTML: &str =
    "Generate an output HTML page with charts and tables.";
pub const MANDATORY_TODO_OPEN_HTML: &str = "Open the generated output HTML page in the browser.";
pub const GENERIC_STARTER_TODO: &str = "Understand and complete the user request.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl TodoStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for TodoStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TodoStatus {
    type Err = TodoError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "pending" => Ok(Self::Pending),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(TodoError::Parse(format!(
                "unknown todo status `{other}`; expected pending, in_progress, completed, or failed"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub status: TodoStatus,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoList {
    pub items: Vec<TodoItem>,
}

impl TodoList {
    pub fn initialize(items: &[String]) -> Result<Self, TodoError> {
        let items = with_mandatory_tail(items)?;
        let items = items
            .into_iter()
            .map(|text| TodoItem {
                status: TodoStatus::Pending,
                text,
            })
            .collect();
        let todo_list = Self { items };
        todo_list.validate()?;
        Ok(todo_list)
    }

    pub fn parse(input: &str) -> Result<Self, TodoError> {
        let mut items = Vec::new();
        for (line_number, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            let (index_text, remainder) = line.split_once(". ").ok_or_else(|| {
                TodoError::Parse(format!("invalid todo line {}: `{line}`", line_number + 1))
            })?;
            let expected_index = items.len() + 1;
            let parsed_index = index_text.parse::<usize>().map_err(|_| {
                TodoError::Parse(format!(
                    "invalid todo item number `{index_text}` on line {}",
                    line_number + 1
                ))
            })?;
            if parsed_index != expected_index {
                return Err(TodoError::Parse(format!(
                    "expected todo item {} but found {} on line {}",
                    expected_index,
                    parsed_index,
                    line_number + 1
                )));
            }

            let status_end = remainder.find(']').ok_or_else(|| {
                TodoError::Parse(format!(
                    "todo line {} is missing a closing status bracket",
                    line_number + 1
                ))
            })?;
            let status_text = remainder
                .strip_prefix('[')
                .and_then(|value| value.get(..status_end.saturating_sub(1)))
                .ok_or_else(|| {
                    TodoError::Parse(format!(
                        "todo line {} must start with `[status]`",
                        line_number + 1
                    ))
                })?;
            let text = remainder
                .get(status_end + 1..)
                .map(str::trim)
                .ok_or_else(|| {
                    TodoError::Parse(format!(
                        "todo line {} is missing todo text",
                        line_number + 1
                    ))
                })?;
            if text.is_empty() {
                return Err(TodoError::Parse(format!(
                    "todo line {} has empty todo text",
                    line_number + 1
                )));
            }

            items.push(TodoItem {
                status: status_text.parse()?,
                text: text.to_owned(),
            });
        }

        let todo_list = Self { items };
        todo_list.validate()?;
        Ok(todo_list)
    }

    pub fn load_from_path(path: &Path) -> Result<Self, TodoError> {
        let raw = fs::read_to_string(path).map_err(|source| TodoError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&raw)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), TodoError> {
        self.validate()?;
        fs::write(path, self.render()).map_err(|source| TodoError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn render(&self) -> String {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| format!("{}. [{}] {}", index + 1, item.status, item.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn next_actionable(&self) -> Option<(usize, &TodoItem)> {
        self.items
            .iter()
            .enumerate()
            .find(|(_, item)| item.status != TodoStatus::Completed)
            .map(|(index, item)| (index + 1, item))
    }

    pub fn set_status(&mut self, item_index: usize, status: TodoStatus) -> Result<(), TodoError> {
        if status == TodoStatus::Pending {
            return Err(TodoError::Validation(
                "todo status cannot be set back to pending".to_owned(),
            ));
        }
        let zero_based = item_index.checked_sub(1).ok_or_else(|| {
            TodoError::Validation("todo item index must be at least 1".to_owned())
        })?;
        if zero_based >= self.items.len() {
            return Err(TodoError::Validation(format!(
                "todo item {} does not exist",
                item_index
            )));
        }
        let current = self.items[zero_based].status;
        if current == status {
            self.validate()?;
            return Ok(());
        }

        let next_actionable = self
            .next_actionable()
            .map(|(index, _)| index)
            .ok_or_else(|| {
                TodoError::Validation("all todo items are already completed".to_owned())
            })?;
        if next_actionable != item_index {
            return Err(TodoError::Validation(format!(
                "todo item {} cannot change status before item {} is resolved",
                item_index, next_actionable
            )));
        }

        let valid = matches!(
            (current, status),
            (TodoStatus::Pending, TodoStatus::InProgress)
                | (TodoStatus::Failed, TodoStatus::InProgress)
                | (TodoStatus::InProgress, TodoStatus::Completed)
                | (TodoStatus::InProgress, TodoStatus::Failed)
        );
        if !valid {
            return Err(TodoError::Validation(format!(
                "invalid todo transition {} -> {} for item {}",
                current, status, item_index
            )));
        }
        if status == TodoStatus::Completed && self.items[zero_based].text == GENERIC_STARTER_TODO {
            return Err(TodoError::Validation(
                "the generic starter todo cannot be completed; replace it with a concrete replan first".to_owned(),
            ));
        }

        self.items[zero_based].status = status;
        self.validate()
    }

    pub fn replan_pending_suffix(&mut self, items: &[String]) -> Result<(), TodoError> {
        let preserve_until = self.replan_start_index()?;
        let mut preserved = self.items[..preserve_until].to_vec();
        let preserved_texts = preserved
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>();
        let sanitized_items = items
            .iter()
            .filter_map(|text| sanitize_replan_item(text, &preserved_texts))
            .collect::<Vec<_>>();
        let replacement = with_mandatory_tail(&sanitized_items)?
            .into_iter()
            .map(|text| TodoItem {
                status: TodoStatus::Pending,
                text,
            })
            .collect::<Vec<_>>();
        preserved.extend(replacement);
        self.items = preserved;
        self.validate()
    }

    fn replan_start_index(&self) -> Result<usize, TodoError> {
        let Some((next_actionable_index, next_actionable)) = self.next_actionable() else {
            return Err(TodoError::Validation(
                "cannot replan because the todo list has no pending suffix".to_owned(),
            ));
        };
        if next_actionable.text == GENERIC_STARTER_TODO {
            return Ok(next_actionable_index - 1);
        }
        if self.items.iter().any(|item| item.status == TodoStatus::Failed) {
            return self
                .items
                .iter()
                .position(|item| item.status == TodoStatus::Pending)
                .ok_or_else(|| {
                    TodoError::Validation(
                        "cannot replan because the todo list has no pending suffix".to_owned(),
                    )
                });
        }
        Err(TodoError::Validation(
            "cannot replan a concrete todo plan after execution has started unless a todo has failed".to_owned(),
        ))
    }

    pub fn validate(&self) -> Result<(), TodoError> {
        let mut phase = TodoValidationPhase::CompletedPrefix;
        for (index, item) in self.items.iter().enumerate() {
            match phase {
                TodoValidationPhase::CompletedPrefix => match item.status {
                    TodoStatus::Completed => {}
                    TodoStatus::Pending => phase = TodoValidationPhase::PendingSuffix,
                    TodoStatus::InProgress | TodoStatus::Failed => {
                        phase = TodoValidationPhase::ActiveThenPending
                    }
                },
                TodoValidationPhase::ActiveThenPending => match item.status {
                    TodoStatus::Pending => phase = TodoValidationPhase::PendingSuffix,
                    other => {
                        return Err(TodoError::Validation(format!(
                            "todo item {} has invalid status {}; after the first in_progress or failed item only pending items may follow",
                            index + 1,
                            other
                        )));
                    }
                },
                TodoValidationPhase::PendingSuffix => {
                    if item.status != TodoStatus::Pending {
                        return Err(TodoError::Validation(format!(
                            "todo item {} has invalid status {}; after the first pending item all remaining items must stay pending",
                            index + 1,
                            item.status
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TodoValidationPhase {
    CompletedPrefix,
    ActiveThenPending,
    PendingSuffix,
}

fn with_mandatory_tail(items: &[String]) -> Result<Vec<String>, TodoError> {
    let mut normalized = Vec::new();
    for item in items {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            return Err(TodoError::Validation(
                "todo items cannot be blank".to_owned(),
            ));
        }
        if canonical_mandatory_todo(trimmed).is_some() {
            continue;
        }
        normalized.push(trimmed.to_owned());
    }
    normalized.push(MANDATORY_TODO_GENERATE_HTML.to_owned());
    normalized.push(MANDATORY_TODO_OPEN_HTML.to_owned());
    Ok(normalized)
}

fn sanitize_replan_item(item: &str, preserved_texts: &[&str]) -> Option<String> {
    let mut text = item.trim();
    if let Some(stripped) = strip_rendered_todo_prefix(text) {
        text = stripped;
    }
    let trimmed = text.trim();
    if trimmed.is_empty()
        || canonical_mandatory_todo(trimmed).is_some()
        || preserved_texts.iter().any(|existing| *existing == trimmed)
    {
        return None;
    }
    Some(trimmed.to_owned())
}

fn strip_rendered_todo_prefix(value: &str) -> Option<&str> {
    let (number, remainder) = value.split_once(". ")?;
    if number.parse::<usize>().is_err() {
        return None;
    }
    if let Some(without_status) = remainder
        .strip_prefix("[pending] ")
        .or_else(|| remainder.strip_prefix("[in_progress] "))
        .or_else(|| remainder.strip_prefix("[completed] "))
        .or_else(|| remainder.strip_prefix("[failed] "))
    {
        return Some(without_status);
    }
    Some(remainder)
}

fn canonical_mandatory_todo(value: &str) -> Option<&'static str> {
    let normalized = normalize_for_match(value);
    let generate_verbs = ["generate", "create", "build", "render", "produce", "write"];
    let mentions_html_report =
        normalized.contains("html") || normalized.contains("report") || normalized.contains("page");
    let mentions_visual_payload =
        normalized.contains("chart") || normalized.contains("table") || normalized.contains("visual");

    if normalized.contains("open")
        && (normalized.contains("html")
            || normalized.contains("browser")
            || normalized.contains("report")
            || normalized.contains("page"))
    {
        return Some(MANDATORY_TODO_OPEN_HTML);
    }

    if generate_verbs.iter().any(|verb| normalized.contains(verb))
        && mentions_html_report
        && mentions_visual_payload
    {
        return Some(MANDATORY_TODO_GENERATE_HTML);
    }

    None
}

fn normalize_for_match(value: &str) -> String {
    value
        .chars()
        .map(|char| match char {
            'A'..='Z' => char.to_ascii_lowercase(),
            'a'..='z' | '0'..='9' => char,
            _ => ' ',
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum TodoError {
    #[error("todo parse error: {0}")]
    Parse(String),
    #[error("todo validation error: {0}")]
    Validation(String),
    #[error("failed to read todo file `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write todo file `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        GENERIC_STARTER_TODO, MANDATORY_TODO_GENERATE_HTML, MANDATORY_TODO_OPEN_HTML, TodoError,
        TodoList, TodoStatus,
    };

    #[test]
    fn initialize_sets_all_items_to_pending_and_appends_mandatory_tail() {
        let todos = TodoList::initialize(&[
            "Inspect the data".to_owned(),
            MANDATORY_TODO_GENERATE_HTML.to_owned(),
        ])
        .expect("initialize should succeed");

        assert_eq!(todos.items.len(), 3);
        assert_eq!(todos.items[0].status, TodoStatus::Pending);
        assert_eq!(todos.items[1].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(todos.items[2].text, MANDATORY_TODO_OPEN_HTML);
    }

    #[test]
    fn parse_rejects_non_sequential_indices() {
        let error = TodoList::parse("2. [pending] Inspect\n")
            .expect_err("non-sequential numbering should fail");
        assert!(matches!(error, TodoError::Parse(_)));
    }

    #[test]
    fn validate_rejects_completed_after_pending() {
        let error =
            TodoList::parse("1. [pending] Inspect\n2. [completed] Analyze\n3. [pending] Render\n")
                .expect_err("completed item cannot appear after pending");
        assert!(matches!(error, TodoError::Validation(_)));
    }

    #[test]
    fn set_status_enforces_ordered_progression() {
        let mut todos = TodoList::initialize(&["Inspect".to_owned(), "Analyze".to_owned()])
            .expect("initialize should succeed");
        todos
            .set_status(1, TodoStatus::InProgress)
            .expect("pending -> in_progress should succeed");
        todos
            .set_status(1, TodoStatus::Completed)
            .expect("in_progress -> completed should succeed");
        let error = todos
            .set_status(3, TodoStatus::InProgress)
            .expect_err("cannot skip item 2");
        assert!(error.to_string().contains("before item 2"));
    }

    #[test]
    fn set_status_is_idempotent_for_same_status() {
        let mut todos = TodoList::initialize(&["Inspect".to_owned(), "Analyze".to_owned()])
            .expect("initialize should succeed");
        todos
            .set_status(1, TodoStatus::InProgress)
            .expect("pending -> in_progress should succeed");
        todos
            .set_status(1, TodoStatus::InProgress)
            .expect("repeating in_progress should be a no-op");
        todos
            .set_status(1, TodoStatus::Completed)
            .expect("in_progress -> completed should succeed");
        todos
            .set_status(1, TodoStatus::Completed)
            .expect("repeating completed should be a no-op");
    }

    #[test]
    fn generic_starter_cannot_be_completed_without_replanning() {
        let mut todos =
            TodoList::initialize(&[GENERIC_STARTER_TODO.to_owned()]).expect("init should work");
        todos
            .set_status(1, TodoStatus::InProgress)
            .expect("starter can enter in_progress while waiting for replan");
        let error = todos
            .set_status(1, TodoStatus::Completed)
            .expect_err("generic starter must not be completable");
        assert!(error
            .to_string()
            .contains("generic starter todo cannot be completed"));
    }

    #[test]
    fn replan_rewrites_only_pending_suffix() {
        let mut todos = TodoList::parse(
            "1. [completed] Inspect\n2. [failed] Analyze\n3. [pending] Summarize\n4. [pending] Generate an output HTML page with charts and tables.\n5. [pending] Open the generated output HTML page in the browser.\n",
        )
        .expect("seed todos should parse");

        todos
            .replan_pending_suffix(&["Retry analysis with grouping".to_owned()])
            .expect("replan should succeed");

        assert_eq!(todos.items[0].status, TodoStatus::Completed);
        assert_eq!(todos.items[1].status, TodoStatus::Failed);
        assert_eq!(todos.items[2].text, "Retry analysis with grouping");
        assert_eq!(todos.items[3].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(todos.items[4].text, MANDATORY_TODO_OPEN_HTML);
    }

    #[test]
    fn replan_replaces_generic_starter_when_it_is_current_actionable() {
        let mut todos = TodoList::parse(
            "1. [in_progress] Understand and complete the user request.\n2. [pending] Generate an output HTML page with charts and tables.\n3. [pending] Open the generated output HTML page in the browser.\n",
        )
        .expect("seed todos should parse");

        todos
            .replan_pending_suffix(&[
                "Define the CSK IPL 2025 analysis scope".to_owned(),
                "Collect and prepare CSK IPL 2025 season data".to_owned(),
            ])
            .expect("replan should succeed");

        assert_eq!(todos.items[0].status, TodoStatus::Pending);
        assert_eq!(todos.items[0].text, "Define the CSK IPL 2025 analysis scope");
        assert_eq!(
            todos.items[1].text,
            "Collect and prepare CSK IPL 2025 season data"
        );
        assert_eq!(todos.items[2].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(todos.items[3].text, MANDATORY_TODO_OPEN_HTML);
        assert!(
            todos
                .items
                .iter()
                .all(|item| item.text != GENERIC_STARTER_TODO)
        );
    }

    #[test]
    fn initialize_dedupes_equivalent_html_tail_aliases() {
        let todos = TodoList::initialize(&[
            "Define scope".to_owned(),
            "Generate the output HTML page with analysis, charts, and tables".to_owned(),
            "Open the generated HTML report in the browser".to_owned(),
            "Generate a final consolidated HTML summary page with charts and tables".to_owned(),
        ])
        .expect("initialize should succeed");

        assert_eq!(todos.items.len(), 3);
        assert_eq!(todos.items[0].text, "Define scope");
        assert_eq!(todos.items[1].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(todos.items[2].text, MANDATORY_TODO_OPEN_HTML);
    }

    #[test]
    fn replan_strips_rendered_prefixes_and_drops_preserved_duplicates() {
        let mut todos = TodoList::parse(
            "1. [completed] Define scope\n2. [failed] Collect data\n3. [pending] Analyze\n4. [pending] Generate an output HTML page with charts and tables.\n5. [pending] Open the generated output HTML page in the browser.\n",
        )
        .expect("seed todos should parse");

        todos
            .replan_pending_suffix(&[
                "1. [completed] Define scope".to_owned(),
                "2. [failed] Collect data".to_owned(),
                "3. [pending] Build local analysis files".to_owned(),
            ])
            .expect("replan should succeed");

        assert_eq!(todos.items[0].text, "Define scope");
        assert_eq!(todos.items[1].text, "Collect data");
        assert_eq!(todos.items[2].text, "Build local analysis files");
        assert_eq!(todos.items[3].text, MANDATORY_TODO_GENERATE_HTML);
        assert_eq!(todos.items[4].text, MANDATORY_TODO_OPEN_HTML);
    }

    #[test]
    fn replan_rejects_concrete_plan_without_failed_todo() {
        let mut todos = TodoList::parse(
            "1. [completed] Define scope\n2. [pending] Collect data\n3. [pending] Generate an output HTML page with charts and tables.\n4. [pending] Open the generated output HTML page in the browser.\n",
        )
        .expect("seed todos should parse");

        let error = todos
            .replan_pending_suffix(&["Compute batting metrics".to_owned()])
            .expect_err("concrete plan should not be replanned without failure");
        assert!(error
            .to_string()
            .contains("cannot replan a concrete todo plan after execution has started"));
    }
}
