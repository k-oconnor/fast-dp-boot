"""Compare against the pinned authors' actual Python quadrature implementation.

AST extraction avoids running their command-line plotting/experiment side effects.
The numerical function and density-building statements execute unchanged.
Upstream source and Apache-2.0 license live in reference/upstream.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import time
import warnings
import numpy as np
import scipy
from scipy.stats import binom, norm
from dp_boot import bootstrap_privacy_profile


def load_reference(path, geometry=False):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    fn = functions["approx_f_numerical"]
    outer_loop = functions["dpboot_dual_all"].body[0]
    body = outer_loop.body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "mu")
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "run_expr_dual")
    factory = ast.parse("def make_density(n, num_composition, epsilon_bin):\n    pass").body[0]
    return_source = "return dens_func_Q, log_likelihood_ratio_func, locals()" if geometry else "return dens_func_Q, log_likelihood_ratio_func"
    factory.body = body[start:end] + ast.parse(return_source).body
    module = ast.fix_missing_locations(ast.Module(body=[fn, factory], type_ignores=[]))
    scope = dict(np=np, scipy=scipy, binom=binom, norm=norm)
    exec(compile(module, str(path), "exec"), scope)
    return scope["approx_f_numerical"], scope["make_density"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--bins", type=int, default=81)
    parser.add_argument("--output", default="benchmarks/upstream_accounting.json")
    args = parser.parse_args()
    path = Path("reference/upstream/dpboot_composition.py")
    reference, make_density = load_reference(path)
    epsilon = np.linspace(-6, 10, args.bins)
    start = time.perf_counter()
    q, logratio = make_density(args.n, args.replicates, args.bins)
    with warnings.catch_warnings(record=True) as caught:
        _, _, delta = reference(q, logratio, args.replicates, epsilon,
                                alpha=np.array([.5]))
    upstream_seconds = time.perf_counter()-start
    mask = epsilon >= 0
    profiles = []
    for step in [.002, .001]:
        start = time.perf_counter()
        profile = bootstrap_privacy_profile(args.n, 1/np.sqrt(args.replicates),
                                            args.replicates, epsilon[mask],
                                            loss_step=step, max_loss=12.)
        elapsed = time.perf_counter()-start
        profiles.append(dict(loss_step=step, seconds=elapsed,
                             speedup=upstream_seconds/elapsed,
                             max_absolute_delta_difference=float(np.max(np.abs(profile.delta-delta[mask]))),
                             tail_bound=profile.discarded_tail_bound,
                             numerical_mass_error=profile.numerical_mass_error,
                             delta=profile.delta.tolist()))
    result = dict(upstream_commit=subprocess.check_output(
        ["git", "-C", "reference/upstream", "rev-parse", "HEAD"], text=True).strip(),
        n=args.n, replicates=args.replicates, upstream_bins=args.bins,
        upstream_seconds=upstream_seconds, upstream_warnings=[str(w.message) for w in caught],
        epsilon=epsilon[mask].tolist(), upstream_delta=delta[mask].tolist(), fft=profiles,
        notes="One timing per method, including density construction. Different numerical "
              "discretizations; speedups must be read together with profile discrepancies. "
              "Unmodified upstream recursion uses quad's default infinite integration limits.")
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ['epsilon','upstream_delta','fft']}, indent=2))
    for p in profiles:
        print({k:v for k,v in p.items() if k != "delta"})


if __name__ == "__main__":
    main()
