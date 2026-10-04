from framework.evolution.config import CustomerSearchConfig, SplitConfig
from framework.evolution.customer.runner import CustomerEvolutionRunner
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.scenario_cases import get_case_provider
from framework.evolution.split_manager import SplitManager


EXPECTED_PATH_COUNTS = {
    "ecommerce_refund": 15,
    "telecom_package": 26,
    "property_service": 23,
    "logistics_delivery": 18,
    "airline_refund": 19,
    "online_education": 21,
}


def test_every_registered_scenario_builds_cases_for_all_mapped_paths():
    for scenario, expected_count in EXPECTED_PATH_COUNTS.items():
        provider = get_case_provider(scenario)
        paths = provider.generate_paths()
        mapping = provider.intent_path_mapping()
        mapped_ids = {
            path_id
            for intent_data in mapping.values()
            for path_id in intent_data.get("possible_paths", [])
        }
        assert len(paths) == expected_count
        assert mapped_ids == set(range(1, expected_count + 1))

        splits = SplitManager(SplitConfig(seed=23), scenario=scenario).build()
        assert len(splits.all_cases) == expected_count
        assert {case.path_id for case in splits.all_cases} == mapped_ids
        assert {case.case_spec["scenario"] for case in splits.all_cases} == {scenario}
        assert {case.intent for case in splits.all_cases} <= set(mapping)


def test_split_manager_rejects_unknown_scenario_instead_of_falling_back():
    try:
        SplitManager(scenario="not_a_scenario").build()
    except ValueError as exc:
        assert "no scenario case provider" in str(exc)
    else:
        raise AssertionError("unknown scenario must not silently use Ecommerce PathList")


def test_customer_search_runner_passes_configured_scenario_to_split_manager(tmp_path):
    config = CustomerSearchConfig(
        scenario="telecom_package",
        max_generations=1,
        splits=SplitConfig(seed=23, max_cases=3),
    )
    runner = CustomerEvolutionRunner(
        config, evaluator=MockEpisodeEvaluator(), run_dir=tmp_path / "telecom-run",
    )
    assert runner.split_manager.scenario == "telecom_package"
    runner.run()
    loaded = runner.split_manager.load(runner.store.run_dir / "split_manifest")
    assert {case.case_spec["scenario"] for case in loaded.all_cases} == {"telecom_package"}
