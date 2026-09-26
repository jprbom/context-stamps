"""Conservative exact AST decontamination; no semantic/pretraining claim."""

import ast
import gzip
import hashlib
import json

DATASET_SHA = '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101'


def normalized_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def against_humaneval(rows, path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA:
        raise ValueError('Pinned held-out source required for decontamination')
    reference = [json.loads(s) for s in gzip.decompress(raw).splitlines()]
    keys = {normalized_ast(r['prompt'] + r['canonical_solution']) for r in reference}
    overlap = [r['task_id'] for r in rows if normalized_ast(r['code']) in keys]
    return {'dataset_sha256': DATASET_SHA, 'excluded_task_ids': overlap,
            'method': 'Full AST equality after removing docstrings; preserves names, imports and computation',
            'limitations': 'Does not detect renamed, structurally different or semantic equivalents; pretraining contamination unknown',
            'heldout_reference_use': 'Decontamination only; no held-out labels, tests or solutions enter training'}
