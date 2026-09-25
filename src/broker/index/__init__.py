"""Codebase index: tree-sitter symbols and edges per repository, in SQLite.

Built only by ``python -m broker.index <repo>``. Read at spawn by the session
broker to ground the opening prompt in real code. Imports nothing from the
broker's runtime surfaces (enforced by import-linter).

The retrieval surface the session broker consumes:
``retrieve`` → ``GroundingContext`` → ``fit_to_budget`` → ``render_relevant_code``.

Terms — one word, one concept:
- symbol: any indexed node (function, method, class, type).
- member: a symbol inside a class body (method or nested class).
- imported_name: the name an import takes from a module, vs local_name.
- member_access_node: AST node type for dotted access a.b.c.
"""
