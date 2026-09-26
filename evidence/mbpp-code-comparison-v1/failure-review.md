# Review of every newly failing task

By Prashant Jagtap

The base passes each task below; the adapter fails it in the frozen comparison.
These are source-level diagnoses from the retained generated code and published
task specification, checked against the native outcomes. They are not proof of
which training example or optimization choice caused a regression. Programs were
not executed on the host during this review.

| HumanEval ID | Observable change in the adapter output |
|---|---|
| 5 | Calls `zip_longest` and `repeat` without importing them; the iterator construction also departs from the requested finite interspersing operation. |
| 16 | The parameter `string` shadows the imported module, then the code accesses `string.punctuation`; case-insensitive distinct counting is also lost. |
| 23 | Counts regex word characters instead of all characters. Base tests pass; additional tests reject it. |
| 43 | Emits a lambda assignment instead of the required function definition and applies `&` to generator expressions. Fails the registered format check. |
| 54 | Compares unrelated regex-match counts instead of character sets. |
| 55 | Replaces iterative Fibonacci with naive recursion. Base tests pass; an additional check fails after about four seconds, consistent with a complexity regression. The worker does not retain the per-input exception, so timeout attribution remains an inference. |
| 56 | Emits a lambda named `correct` instead of `correct_bracketing`; the regex does not implement general nested-bracket matching. Fails the format check. |
| 57 | Checks increasing order only; the specification also accepts decreasing order. |
| 68 | Returns `[-1, -1]` when a nonempty input has no even value; the specified result is an empty list. |
| 69 | Returns the first qualifying value in ascending order rather than the greatest qualifying value. |
| 74 | Adds both list lengths together instead of comparing their separate totals. |
| 81 | Omits the `D-` interval for GPAs above zero and at most 0.7. |
| 104 | Returns combinations of qualifying values instead of a sorted list of the qualifying values themselves. |
| 106 | Returns the even index itself instead of its factorial. |
| 116 | Sorts numerically while ignoring the required binary-popcount primary key. |
| 117 | Filters by word length rather than by the number of consonants. |
| 123 | Collects even sequence values and omits the initial value; the task requests odd Collatz values including the initial value when applicable. |
| 157 | Assumes the third argument is the hypotenuse instead of considering the possible side orderings. |
| 159 | Uses a quotient/remainder formula unrelated to the specified available-stock update; it can divide by zero for allowed inputs. |

Most newly failing programs still have a valid function-shaped response. Reducing
format failures therefore did not preserve semantics. The 20-to-5 format
improvement cannot be presented as a correctness gain. The additional-test stage
also finds more failures in the adapter arm (nine versus four across all tasks).

## What this changes about the next experiment

1. Keep the immutable base as the active control. Do not use benchmark task IDs,
   known pass/fail labels or hidden tests to choose an adapter at inference time.
2. Test a conservative update on independent training material: smaller changes
   and a base-distribution penalty are candidates for limiting forgetting.
   Such a penalty may preserve base errors too; it needs measured comparison,
   not an assumption that a mathematical regularizer guarantees retention.
3. Include independently checked imports, entry-point contracts, boundary cases
   and complexity in training admission and fresh validation where available.
   Passing three example assertions is insufficient to establish those properties.
4. Reserve a fresh task partition before selecting the next recipe. Keep this
   HumanEval run as a development/regression record after inspecting its failures.
   Measure new failures and the complete inference/verification cost, not only
   average loss or aggregate score.
5. Evaluate verified memory/context selection separately on the same frozen
   model. This comparison did not supply or test missing context, so its failure
   neither proves nor disproves a benefit from the Context Stamps runtime.

These are next experimental changes, not fixes already demonstrated by this run.
