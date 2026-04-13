"""Parse raw AMR text files into pandas DataFrames."""
import contextlib
import io
import re

import pandas as pd
import penman
from penman.models.noop import NoOpModel


def penman_to_one_line(penman_str: str) -> str:
    lines = penman_str.strip().split('\n')
    one_line = ' '.join(line.strip() for line in lines)
    return re.sub(r'\s+', ' ', one_line)


def fix_missing_closing_brackets(graph_str: str) -> str:
    missing = graph_str.count('(') - graph_str.count(')')
    if missing > 0:
        graph_str += ')' * missing
    return graph_str


def fix_multiword_nodes(graph_str: str) -> str:
    def repl(match):
        return '/ ' + match.group(1).replace(' ', '_')
    return re.sub(r'/ ([^\(\):]+)', repl, graph_str)


def decode_with_warnings(graph_str: str, sent: str):
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        try:
            graph = penman.decode(graph_str, model=NoOpModel())
            warnings = f.getvalue()
            if warnings.strip():
                print(f"Warning(s) during decoding sentence: {sent}")
                print(warnings)
            return graph, None
        except Exception as e:
            return None, e


def read_amr_direct(filename: str, one_line: bool = True) -> pd.DataFrame:
    """Read an AMR file into a DataFrame with 'query' and 'amr' columns."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    queries, amr_list = [], []
    current_sent = None
    current_graph_lines: list[str] = []

    def flush():
        if current_sent is None or not current_graph_lines:
            return
        graph_str = "\n".join(current_graph_lines).strip()
        graph_str = fix_missing_closing_brackets(graph_str)
        graph_str = fix_multiword_nodes(graph_str)
        graph, error = decode_with_warnings(graph_str, current_sent)
        if error:
            return
        amr_str = penman.encode(graph, model=NoOpModel())
        if one_line:
            amr_str = penman_to_one_line(amr_str)
        queries.append(current_sent)
        amr_list.append(amr_str)

    for line in lines:
        line = line.strip()
        if line.startswith("#::snt"):
            flush()
            current_graph_lines = []
            current_sent = line[len("#::snt"):].strip()
        elif line == "":
            continue
        else:
            current_graph_lines.append(line)
    flush()

    return pd.DataFrame({"query": queries, "amr": amr_list})
