//! Shared MCP metadata artifact parsing and rendering.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const CURRENT_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_METADATA_DIR: &str = ".arka/mcp";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpArtifactPaths {
    pub dir: PathBuf,
    pub minimal_path: PathBuf,
    pub full_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    Tool,
    Resource,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerMetadata {
    pub logical_name: String,
    #[serde(default)]
    pub protocol_name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub instructions_summary: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCapabilityFamilySummary {
    pub supported: bool,
    pub count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCapabilityFamilies {
    pub tools: McpCapabilityFamilySummary,
    pub resources: McpCapabilityFamilySummary,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MinimalToolMetadata {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FullToolMetadata {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MinimalResourceMetadata {
    pub uri: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FullResourceMetadata {
    pub uri: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub annotations: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpMinimalCatalog {
    pub schema_version: u32,
    pub server: McpServerMetadata,
    pub capability_families: McpCapabilityFamilies,
    #[serde(default)]
    pub tools: Vec<MinimalToolMetadata>,
    #[serde(default)]
    pub resources: Vec<MinimalResourceMetadata>,
}

impl McpMinimalCatalog {
    pub fn validate(&self) -> Result<(), MetadataError> {
        if self.server.logical_name.trim().is_empty() {
            return Err(MetadataError::Invalid(
                "minimal catalog server.logical_name cannot be blank".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpFullCatalog {
    pub schema_version: u32,
    pub server: McpServerMetadata,
    pub capability_families: McpCapabilityFamilies,
    #[serde(default)]
    pub tools: Vec<FullToolMetadata>,
    #[serde(default)]
    pub resources: Vec<FullResourceMetadata>,
    #[serde(default)]
    pub extensions: Value,
}

impl McpFullCatalog {
    pub fn validate(&self) -> Result<(), MetadataError> {
        if self.server.logical_name.trim().is_empty() {
            return Err(MetadataError::Invalid(
                "full catalog server.logical_name cannot be blank".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("failed to read metadata file `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write metadata file `{path}`: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("metadata file `{path}` is missing YAML front matter")]
    MissingFrontMatter { path: PathBuf },
    #[error("failed to parse metadata front matter in `{path}`: {source}")]
    FrontMatter {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("{0}")]
    Invalid(String),
}

pub fn artifact_paths(base_dir: &Path, logical_name: &str) -> McpArtifactPaths {
    let dir = if base_dir.as_os_str().is_empty() {
        PathBuf::from(DEFAULT_METADATA_DIR)
    } else {
        base_dir.to_path_buf()
    };
    McpArtifactPaths {
        minimal_path: dir.join(format!("{logical_name}_minimal.md")),
        full_path: dir.join(format!("{logical_name}_full.md")),
        dir,
    }
}

pub fn write_catalogs(
    paths: &McpArtifactPaths,
    minimal: &McpMinimalCatalog,
    full: &McpFullCatalog,
) -> Result<(), MetadataError> {
    minimal.validate()?;
    full.validate()?;
    fs::create_dir_all(&paths.dir).map_err(|source| MetadataError::Write {
        path: paths.dir.clone(),
        source,
    })?;
    atomic_write(
        &paths.minimal_path,
        &render_minimal_catalog_markdown(minimal),
    )?;
    atomic_write(&paths.full_path, &render_full_catalog_markdown(full))?;
    Ok(())
}

pub fn load_minimal_catalog(path: &Path) -> Result<McpMinimalCatalog, MetadataError> {
    let raw = fs::read_to_string(path).map_err(|source| MetadataError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let front_matter = extract_front_matter(path, &raw)?;
    let parsed = serde_yaml::from_str::<McpMinimalCatalog>(&front_matter).map_err(|source| {
        MetadataError::FrontMatter {
            path: path.to_path_buf(),
            source,
        }
    })?;
    parsed.validate()?;
    Ok(parsed)
}

pub fn load_full_catalog(path: &Path) -> Result<McpFullCatalog, MetadataError> {
    let raw = fs::read_to_string(path).map_err(|source| MetadataError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let front_matter = extract_front_matter(path, &raw)?;
    let parsed = serde_yaml::from_str::<McpFullCatalog>(&front_matter).map_err(|source| {
        MetadataError::FrontMatter {
            path: path.to_path_buf(),
            source,
        }
    })?;
    parsed.validate()?;
    Ok(parsed)
}

pub fn render_minimal_catalog_markdown(catalog: &McpMinimalCatalog) -> String {
    let mut markdown = render_front_matter(catalog);
    markdown.push_str(&format!(
        "# MCP Minimal: {}\n\n",
        catalog.server.logical_name
    ));
    markdown.push_str("## Server\n");
    markdown.push_str(&render_server_markdown(&catalog.server));
    markdown.push_str("\n## Tools\n");
    if catalog.tools.is_empty() {
        markdown.push_str("- None\n");
    } else {
        for tool in &catalog.tools {
            markdown.push_str(&format!(
                "- `{}`: {}\n",
                tool.name,
                tool.description.as_deref().unwrap_or("No description")
            ));
        }
    }
    markdown.push_str("\n## Resources\n");
    if catalog.resources.is_empty() {
        markdown.push_str("- None\n");
    } else {
        for resource in &catalog.resources {
            markdown.push_str(&format!(
                "- `{}`: {}\n",
                resource.uri,
                resource.description.as_deref().unwrap_or("No description")
            ));
        }
    }
    markdown
}

pub fn render_full_catalog_markdown(catalog: &McpFullCatalog) -> String {
    let mut markdown = render_front_matter(catalog);
    markdown.push_str(&format!("# MCP Full: {}\n\n", catalog.server.logical_name));
    markdown.push_str("## Server\n");
    markdown.push_str(&render_server_markdown(&catalog.server));
    markdown.push_str("\n## Tools\n");
    if catalog.tools.is_empty() {
        markdown.push_str("- None\n");
    } else {
        for tool in &catalog.tools {
            markdown.push_str(&format!(
                "- `{}`: {}\n",
                tool.name,
                tool.description.as_deref().unwrap_or("No description")
            ));
        }
    }
    markdown.push_str("\n## Resources\n");
    if catalog.resources.is_empty() {
        markdown.push_str("- None\n");
    } else {
        for resource in &catalog.resources {
            markdown.push_str(&format!(
                "- `{}`: {}\n",
                resource.uri,
                resource.description.as_deref().unwrap_or("No description")
            ));
        }
    }
    markdown
}

fn render_front_matter<T: Serialize>(value: &T) -> String {
    let yaml = serde_yaml::to_string(value).expect("catalog should serialize");
    format!("---\n{yaml}---\n\n")
}

fn render_server_markdown(server: &McpServerMetadata) -> String {
    let mut lines = String::new();
    lines.push_str(&format!("- Logical name: `{}`\n", server.logical_name));
    if !server.protocol_name.trim().is_empty() {
        lines.push_str(&format!("- Protocol name: `{}`\n", server.protocol_name));
    }
    if let Some(title) = &server.title {
        lines.push_str(&format!("- Title: {}\n", title));
    }
    if !server.version.trim().is_empty() {
        lines.push_str(&format!("- Version: `{}`\n", server.version));
    }
    if let Some(description) = &server.description {
        lines.push_str(&format!("- Description: {}\n", description));
    }
    if let Some(summary) = &server.instructions_summary {
        lines.push_str(&format!("- Instructions summary: {}\n", summary));
    }
    lines
}

fn extract_front_matter(path: &Path, raw: &str) -> Result<String, MetadataError> {
    let mut lines = raw.lines();
    if lines.next() != Some("---") {
        return Err(MetadataError::MissingFrontMatter {
            path: path.to_path_buf(),
        });
    }
    let mut front_matter = String::new();
    for line in lines {
        if line == "---" {
            return Ok(front_matter);
        }
        front_matter.push_str(line);
        front_matter.push('\n');
    }
    Err(MetadataError::MissingFrontMatter {
        path: path.to_path_buf(),
    })
}

fn atomic_write(path: &Path, body: &str) -> Result<(), MetadataError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| MetadataError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(path, body).map_err(|source| MetadataError::Write {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use super::{
        CURRENT_SCHEMA_VERSION, FullResourceMetadata, FullToolMetadata, McpCapabilityFamilies,
        McpCapabilityFamilySummary, McpFullCatalog, McpMinimalCatalog, McpServerMetadata,
        MinimalResourceMetadata, MinimalToolMetadata, artifact_paths, load_full_catalog,
        load_minimal_catalog, write_catalogs,
    };

    #[test]
    fn artifact_paths_use_expected_suffixes() {
        let paths = artifact_paths(Path::new(".arka/mcp"), "crm");
        assert!(paths.minimal_path.ends_with("crm_minimal.md"));
        assert!(paths.full_path.ends_with("crm_full.md"));
    }

    #[test]
    fn catalogs_round_trip_from_markdown_front_matter() {
        let dir = std::env::temp_dir().join(format!("mcp-metadata-test-{}", std::process::id()));
        let paths = artifact_paths(&dir, "crm");
        let families = McpCapabilityFamilies {
            tools: McpCapabilityFamilySummary {
                supported: true,
                count: 1,
            },
            resources: McpCapabilityFamilySummary {
                supported: true,
                count: 1,
            },
        };
        let minimal = McpMinimalCatalog {
            schema_version: CURRENT_SCHEMA_VERSION,
            server: McpServerMetadata {
                logical_name: "crm".to_owned(),
                protocol_name: "fake-crm".to_owned(),
                title: Some("CRM".to_owned()),
                version: "1.0.0".to_owned(),
                description: Some("Customer data".to_owned()),
                instructions_summary: Some("Use read-only flows".to_owned()),
            },
            capability_families: families.clone(),
            tools: vec![MinimalToolMetadata {
                name: "search_contacts".to_owned(),
                title: Some("Search Contacts".to_owned()),
                description: Some("Search contacts".to_owned()),
            }],
            resources: vec![MinimalResourceMetadata {
                uri: "crm://dashboards/main".to_owned(),
                name: Some("main".to_owned()),
                title: Some("Main".to_owned()),
                description: Some("Dashboard".to_owned()),
                mime_type: Some("application/json".to_owned()),
            }],
        };
        let full = McpFullCatalog {
            schema_version: CURRENT_SCHEMA_VERSION,
            server: minimal.server.clone(),
            capability_families: families,
            tools: vec![FullToolMetadata {
                name: "search_contacts".to_owned(),
                title: Some("Search Contacts".to_owned()),
                description: Some("Search contacts".to_owned()),
                input_schema: json!({"type": "object"}),
            }],
            resources: vec![FullResourceMetadata {
                uri: "crm://dashboards/main".to_owned(),
                name: Some("main".to_owned()),
                title: Some("Main".to_owned()),
                description: Some("Dashboard".to_owned()),
                mime_type: Some("application/json".to_owned()),
                annotations: None,
            }],
            extensions: json!({}),
        };

        write_catalogs(&paths, &minimal, &full).expect("catalogs should write");
        let loaded_minimal =
            load_minimal_catalog(&paths.minimal_path).expect("minimal should load");
        let loaded_full = load_full_catalog(&paths.full_path).expect("full should load");
        assert_eq!(loaded_minimal, minimal);
        assert_eq!(loaded_full, full);
        let _ = std::fs::remove_file(paths.minimal_path);
        let _ = std::fs::remove_file(paths.full_path);
        let _ = std::fs::remove_dir(paths.dir);
    }
}
