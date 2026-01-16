import torch
from torch import nn

from utils.counterfactual_utils import interchange_hook


class SequentialModule(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.model(x)


class MLP_scaled_tied(nn.Module):
    """
    Parameter-efficient version of MLP_scaled:
    - 1 init block (input -> 13 embeddings)
    - (K-1) transition steps reusing the same 13 modules (weight tying across time)
    Hidden states are still exposed as a list of length 13*K, so IIT indexing stays the same.
    """
    def __init__(self, input_size: int, output_size: int, scaled_depth: int):
        super().__init__()
        self.scaled_depth = int(scaled_depth)

        # Transition input sizes must match the concatenations in MLP_scaled.
        self.input_size_list = [
            output_size,        # 0
            2 * output_size,    # 1
            2 * output_size,    # 2
            4 * output_size,    # 3
            2 * output_size,    # 4
            4 * output_size,    # 5
            2 * output_size,    # 6
            2 * output_size,    # 7
            2 * output_size,    # 8
            2 * output_size,    # 9
            output_size,        # 10
            2 * output_size,    # 11
            2 * output_size,    # 12
        ]

        # One-time init modules (called once).
        self.init_modules = nn.ModuleList(
            [SequentialModule(input_size, output_size) for _ in range(13)]
        )

        # Reused transition modules (called at every time step > 0).
        self.step_modules = nn.ModuleList(
            [SequentialModule(self.input_size_list[i], output_size) for i in range(13)]
        )

        self.output = nn.Linear(output_size, 1)
        self.loss = nn.MSELoss(reduction="mean")

    def _call_with_optional_intervention(
        self,
        layer_index: int,
        layer_module: nn.Module,
        x: torch.Tensor,
        variable_names,
        interchanged_variables
    ) -> torch.Tensor:
        """
        IMPORTANT:
        Because step_modules are reused across time, we must not leave hooks registered.
        Register -> call once -> remove immediately.
        """
        hooks = []
        if variable_names is not None and layer_index in variable_names:
            assert interchanged_variables is not None
            for interchanged_variable in variable_names[layer_index]:
                interchanged_activations = interchanged_variables[interchanged_variable[0]]
                hooks.append(
                    layer_module.register_forward_hook(
                        interchange_hook(interchanged_variable, interchanged_activations)
                    )
                )

        try:
            out = layer_module(x)
        finally:
            for h in hooks:
                h.remove()

        return out

    def forward(
        self,
        input_ids,
        interchanged_variables=None,
        variable_names=None,
        interchanged_activations=None,  # kept for API compatibility; unused
        t_outputs=None,
        causal_t_outputs=None,
        s_outputs=None,
    ):
        student_output = {"hidden_states": []}

        # Same input preprocessing as other student models.
        x = torch.cat([input_ids[0:10], input_ids[12:]])

        # ---- init step (time_idx = 0) ----
        prev = [None] * 13
        for i in range(13):
            layer_index = i
            h_i = self._call_with_optional_intervention(
                layer_index, self.init_modules[i], x, variable_names, interchanged_variables
            )
            prev[i] = h_i
            student_output["hidden_states"].append(h_i)

        # ---- transition steps (time_idx = 1..K-1) ----
        for t in range(1, self.scaled_depth):
            cur = [None] * 13
            for i in range(13):
                layer_index = t * 13 + i

                # Replicate the parent selection logic of MLP_scaled.
                if i in [0, 10]:
                    inp = prev[i]
                elif i in [1, 2, 4, 6, 8, 11]:
                    inp = torch.cat([prev[i - 1], prev[i]], dim=0)
                elif i == 3:
                    inp = torch.cat([prev[2], prev[3], prev[4], prev[8]], dim=0)
                elif i == 5:
                    inp = torch.cat([prev[5], prev[9], prev[10], prev[11]], dim=0)
                elif i == 7:
                    inp = torch.cat([prev[5], prev[7]], dim=0)
                elif i == 9:
                    inp = torch.cat([prev[5], prev[9]], dim=0)
                elif i == 12:
                    inp = torch.cat([prev[3], prev[12]], dim=0)
                else:
                    raise ValueError(f"Unexpected variable index i={i}")

                h_i = self._call_with_optional_intervention(
                    layer_index, self.step_modules[i], inp, variable_names, interchanged_variables
                )
                cur[i] = h_i
                student_output["hidden_states"].append(h_i)

            prev = cur

        student_output["outputs"] = self.output(student_output["hidden_states"][-1])

        # IIT Objective
        if causal_t_outputs is None:
            # If it is None, it is simply a forward for getting hidden states.
            if t_outputs is not None:
                s_outputs = student_output["outputs"]
                student_output["loss"] = self.loss(s_outputs, t_outputs.unsqueeze(0))
        else:
            # Causal loss.
            causal_s_outputs = student_output["outputs"]
            student_output["loss"] = self.loss(causal_s_outputs, causal_t_outputs.unsqueeze(0))

            # Measure the efficacy of the interchange.
            teacher_interchange_efficacy = (
                self.loss(
                    causal_t_outputs.unsqueeze(0),
                    t_outputs.unsqueeze(0),
                )
            )

            student_interchange_efficacy = (
                self.loss(
                    causal_s_outputs,
                    s_outputs,
                )
            )
            student_output["teacher_interchange_efficacy"] = teacher_interchange_efficacy
            student_output["student_interchange_efficacy"] = student_interchange_efficacy

        return student_output
