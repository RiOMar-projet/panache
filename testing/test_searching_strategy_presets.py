import unittest

from panache.config import build_parameters
from panache.utils import (
    SEARCHING_STRATEGY_PRESETS,
    coordinate_range_bounds,
    define_parameters,
    searching_strategy_directions_from_presets,
)


def minimal_parameters(strategy_name="northward_fan"):
    return {
        "searching_strategies": {"Test plume": strategy_name},
        "bathymetric_threshold": 0,
        "starting_points": {"Test plume": [1.0, 2.0]},
        "core_of_the_plumes": {"Test plume": [1.0, 2.0]},
        "lat_range_of_plume_area": [0.0, 1.0],
        "lon_range_of_plume_area": [0.0, 1.0],
        "threshold_of_cloud_coverage_in_percentage": 25,
        "maximal_bathymetric_for_zone_with_resuspension": {"Test plume": 30},
        "minimal_distance_from_estuary_for_zone_with_resuspension": {"Test plume": 30},
        "max_steps_for_the_directions": {"Test plume": 10},
        "maximal_threshold": {"Test plume": 3},
        "minimal_threshold": {"Test plume": 1},
        "quantile_to_use": {"Test plume": 0.2},
        "fixed_threshold": {"Test plume": 1.5},
        "river_mouth_to_exclude": {},
    }


class SearchingStrategyPresetTests(unittest.TestCase):
    def test_four_cardinal_fan_presets_are_available(self):
        self.assertEqual(
            set(SEARCHING_STRATEGY_PRESETS),
            {"northward_fan", "southward_fan", "eastward_fan", "westward_fan"},
        )

    def test_resolves_named_presets_to_direction_vectors(self):
        directions = searching_strategy_directions_from_presets({"Test plume": "eastward_fan"})

        self.assertEqual(directions["Test plume"], SEARCHING_STRATEGY_PRESETS["eastward_fan"])
        self.assertIsNot(directions["Test plume"], SEARCHING_STRATEGY_PRESETS["eastward_fan"])

    def test_rejects_non_preset_values(self):
        with self.assertRaises(TypeError):
            searching_strategy_directions_from_presets({"Test plume": ["southward_fan"]})

        with self.assertRaises(ValueError):
            searching_strategy_directions_from_presets({"Test plume": "diagonal_fan"})

    def test_coordinate_range_bounds_supports_min_max_and_polygon_lists(self):
        self.assertEqual(coordinate_range_bounds([1.0, 2.0]), (1.0, 2.0))
        self.assertEqual(coordinate_range_bounds([49.75, 49.75, 51.15, 50.4]), (49.75, 51.15))

    def test_build_parameters_accepts_named_presets(self):
        parameters = build_parameters(minimal_parameters("westward_fan"))

        self.assertEqual(parameters["searching_strategies"], {"Test plume": "westward_fan"})
        self.assertEqual(
            parameters["searching_strategy_directions"]["Test plume"],
            SEARCHING_STRATEGY_PRESETS["westward_fan"],
        )

    def test_built_in_zones_use_named_presets(self):
        for zone in [
            "BAY_OF_SEINE",
            "BAY_OF_BISCAY",
            "GULF_OF_LION",
            "SOUTHERN_BRITTANY",
        ]:
            with self.subTest(zone=zone):
                parameters = define_parameters(zone)

                for plume_name, preset_name in parameters["searching_strategies"].items():
                    self.assertIn(preset_name, SEARCHING_STRATEGY_PRESETS)
                    self.assertEqual(
                        parameters["searching_strategy_directions"][plume_name],
                        SEARCHING_STRATEGY_PRESETS[preset_name],
                    )


if __name__ == "__main__":
    unittest.main()
