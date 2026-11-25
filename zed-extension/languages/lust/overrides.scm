; Lust syntax scope overrides for Zed
; Enables `not_in = ["string"]` / `not_in = ["string", "comment"]` in config.toml.

(string) @string
[(comment) (doc_comment)] @comment.inclusive
