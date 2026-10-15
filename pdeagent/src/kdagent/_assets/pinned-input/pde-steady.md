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

When `topology` is `scattered`, the sample points are not a product of the
axes: `n` is the number of sample points and is the same for every axis,
`spacing` is `not_applicable`, the four `step_*` values are null, and the
three diagnostic maps defined below are empty.

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

For a steady input `target` and `lhs_label` are null, `lhs_order` is zero;
the reported field and coordinates specify a homogeneous relation.
