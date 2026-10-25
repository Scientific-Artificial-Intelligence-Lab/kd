## Pinned input glossary

`id` is the data-tool identifier; `task_kind` selects the task instructions;
`target` names the fitted left-hand side; `report` describes the loaded data;
`inventory` describes the source container and its arrays (`key`, `shape`,
`dtype`), or is null for a catalog input. `report.name` is the input name,
`topology` its layout, and `warnings` the inspection warnings.
`fields` lists each field or column's `name`, `dtype`, `shape`, finite-entry
`min`, `max`, `mean`, and counts `nan_count` and `inf_count`.
A null `min`, `max` or `mean` means there were no finite values.

`axes` lists coordinate `name`, point count `n`, finite range `min` and `max`,
`spacing`, `periodic`, and `step_first`, `step_mean`, `step_min`, `step_max`.
`lhs_field`, `lhs_axis`, `lhs_order` and `lhs_label` describe the target.

The field diagnostics follow the definitions in kd's `FieldReport`:
- `noise_estimate`: MAD of the second difference
  `u[i-1] - 2*u[i] + u[i+1]`, scaled by `1.4826 / sqrt(6)` and divided by the
  field's population standard deviation. Even a smooth noiseless field can
  have a nonzero estimate.
- `first_derivative_gap`: `||2C - E|| / ||E||`, with
  `C = u[i+1] - u[i-1]` and `E = u[i+2] - u[i-2]`.
- `second_derivative_gap`: `||4A - B|| / ||B||`, with
  `A = u[i-1] - 2*u[i] + u[i+1]` and `B = u[i-2] - 2*u[i] + u[i+2]`.

Both gaps use the common interior `2 <= i <= n-3`; MAD, standard deviation
and L2 norms pool the other dimensions. Spacing cancels from the ratios.
The diagnostics describe the raw field, independent of the search's derivative
method. An absent axis key means kd did not compute that diagnostic along
that axis: computation needs uniform spacing, at least five points and a
field without NaN or Inf. A null value inside a diagnostic dictionary means
infinity, not a missing measurement.

Derivative source against noise, measured on seven benchmark problems at
relative noise 0, 1e-4, 1e-3, 1e-2 and 5e-2 with three noise draws each. The
true equation refitted on finite differences of the raw field passed (NMSE
below 0.05 and every coefficient within 5%) on all seven without noise, on
two at 1e-4 plus one more on 1 of 3 draws, and on none at 1e-3 and above.
After Savitzky-Golay smoothing with window 11 it passed up to 1e-3 on three
problems and up to 1e-4 on three, and at no level, including no noise, on the
problem with a shock. With DLGA's surrogate derivatives, searches found the
equation on 3 of 3 seeds on two problems at 1e-2. On these fields the median
`noise_estimate` along x was within 5% of the relative noise at 1e-2 and
5e-2, 0.0010 to 0.0018 at 1e-3 on six problems, and 0.0000 to 0.0008 at 1e-4
and without noise on the same six; on the seventh it was 0.0044 to 0.0046 at
every level up to 1e-3.

The evolution target is the named time derivative, including its order.
