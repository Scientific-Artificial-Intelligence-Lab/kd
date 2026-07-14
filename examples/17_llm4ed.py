
import kd
from kd.llm import LLMRequest, LLMResponse
from kd.search.llm4ed.config import Llm4edConfig


class OfflineProvider:

    def prepare(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="<res>u_xx</res>", model="offline", usage=None)



dataset = kd.generate_diffusion_data(
    alpha=1.0, waves=(1.0,), grid_sizes=(64,), nt=40, seed=0
)
print(f"Ground truth: {dataset.ground_truth}")




model = kd.Model(
    algorithm="llm4ed",
    generations=3,
    config=Llm4edConfig(samples_per_epoch=4, max_llm_calls_per_propose=4),
    provider=OfflineProvider(),
)















model.fit(dataset)


print()
print(f"Discovered: {model.best_expr_}")
print(f"Best reward: {model.best_score_:.4f}")
