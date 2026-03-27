//! Strongly typed identifiers used throughout the runtime.
//!
//! These wrappers avoid mixing unrelated IDs in event payloads, state records,
//! and public APIs. They also make it much easier to write narrow tests that
//! assert on a specific lifecycle stage without relying on ad-hoc strings.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! define_id {
    ($name:ident) => {
        #[doc = concat!("Unique identifier for one runtime ", stringify!($name), ".")]
        #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            #[doc = concat!("Generates a new ", stringify!($name), " value.")]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            #[doc = concat!("Returns the inner UUID for this ", stringify!($name), ".")]
            pub fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

define_id!(TurnId);
define_id!(StepId);
define_id!(MessageId);
