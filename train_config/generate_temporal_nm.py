import argparse
import json
from pathlib import Path


def build_mapping(pred_horizon: int, dt: int = 3, n_vars: int = 13):
    assert pred_horizon % dt == 0
    k_steps = pred_horizon // dt
    total = k_steps * n_vars
    mappings = []
    for idx in range(total):
        mappings.append(
            {
                "teacher_variable_names": [f"$L:{idx}$[:,:]"],
                "student_variable_names": [f"$L:{idx}$[:,:]"],
            }
        )
    return {"interchange_variable_mappings": mappings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dt", type=int, default=3)
    args = parser.parse_args()

    data = build_mapping(args.pred_horizon, args.dt)
    Path(args.out).write_text(json.dumps(data, indent=4))
    print(f"Saved {len(data['interchange_variable_mappings'])} mappings to {args.out}")
