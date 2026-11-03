# Local Streamlit runtime and reload safety

NATIP includes a standalone Python interpreter under `python-standalone/`.
That places standard-library files such as `enum.py` and `typing.py` inside
the project folder. Streamlit 1.36's source watcher can treat these as app
modules. On an app edit it removes all watched modules from `sys.modules`,
while third-party packages in `site-packages` remain loaded.

Reloading `enum` or `typing` under those conditions breaks type identity:
Pydantic retains references to the old standard-library types, and new app
classes inherit the new types. Symptoms include `PydanticSchemaGenerationError`
for `DarvasState`, plus warnings that valid `typing.Literal` annotations are
unknown or unvalidated. A new interpreter does not reproduce the error unless
the runtime modules are explicitly unloaded.

The project config disables automatic source reload for normal local use:
`server.fileWatcherType = "none"` and `server.runOnSave = false`.
It also excludes `**/python-standalone`, virtual environments and generated
data folders from watching. Wildcard prefixes are necessary for the installed
Streamlit watcher to match these absolute paths portably.

Restart Streamlit after source edits. Do not fix this runtime issue by enabling
`arbitrary_types_allowed` or replacing validated enums with arbitrary objects.
The normal startup scripts already disable file watching. If enabling polling
for development, keep the runtime folder exclusions in place.

Verification: `tests/test_streamlit_runtime.py` checks the watcher policy,
standard-library exclusions, strict Darvas enum validation and JSON schema
generation. Full-page verification additionally loads the Kronos US selector,
generates a 512-bar forecast from cached adjusted data, and reruns the page.
