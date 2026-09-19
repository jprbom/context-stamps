"""Counterbalanced follow-up after discovering the v1 wording shortcut.

Reuses the frozen v1 evaluator, changes only the disclosed fixture generator and
output directory. v1 results are retained; v2 is not independent of that audit.
"""

import random

import run_spherical as runner


def counterbalanced(split, count, seed):
    rng = random.Random(seed)
    result = []
    tasks = ["calibration", "segmentation", "quantization"] if split == "ood" else ["timeout", "retry", "schema"]
    for i in range(count):
        entity = f"{split}_component_{rng.randrange(100000000)}"
        task = rng.choice(tasks)
        intent = rng.choice(["implement", "review", "measure"])
        wording = [f"Inspect {task} behavior and make required changes",
                   f"Procedure and evidence for {task} modification",
                   f"Record findings from the {task} experiment"]
        query = dict(content=wording[0], entity=entity, intent=intent, task=task)
        candidates = []
        for j in range(8):
            item = dict(query, content=rng.choice(wording))
            if j:
                if j % 3 == 0:
                    item["entity"] = f"{split}_other_{rng.randrange(100000000)}"
                elif j % 3 == 1:
                    item["intent"] = "unrelated_intent_" + str(j)
                else:
                    item["task"] = "unrelated_task_" + str(j)
            candidates.append(item)
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        result.append({"id": f"{split}-{i}", "query": query,
                       "candidates": [candidates[j] for j in indices], "positive": indices.index(0)})
    return result


if __name__ == "__main__":
    runner.OUT = runner.ROOT / "evidence/spherical-v2"
    runner.fixtures = counterbalanced
    runner.main()
