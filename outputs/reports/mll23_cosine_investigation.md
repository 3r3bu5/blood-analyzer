# MLL23 Cosine Head Investigation

Status: `investigated_no_code_defect_confirmed`

The cosine head underperformed the linear and MLP heads on the validation split. Input embeddings and classifier weights are normalized in the cosine-head forward pass, and no label-ordering or evaluation-loader defect was demonstrated by the current tests. The result is treated as inconclusive/expected under this implemented configuration, so the validation-selected MLP remains packaged.
