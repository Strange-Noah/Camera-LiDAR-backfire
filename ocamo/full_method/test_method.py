"""Focused synthetic checks; these cannot certify real-scene convergence."""
import copy
import unittest
import numpy as np
from core import (PaperTracker, automatic_label, camera_points, exp_rotation,
                  extract_corners, project, quality_features, rigid, rotation_error)
from preselection import fit, scores, train_scene_disjoint
from data import perturbation


class Geometry(unittest.TestCase):
    def test_physical_camera_correction_and_moving_pose(self):
        omega = np.array([.01, -.02, .005]); q = exp_rotation(omega)
        points = np.array([[8., 2, 10], [-2., 1, 15]])
        for rotvec in [[0., .2, .1], [.3, -.1, .4]]:
            lc = np.eye(4); lc[:3, :3] = exp_rotation(rotvec); lc[:3, 3] = [.3, -.2, 1.7]
            corrupt = lc.copy(); corrupt[:3, :3] = lc[:3, :3]@q
            np.testing.assert_allclose(camera_points(points, corrupt, -omega),
                                       camera_points(points, lc, np.zeros(3)), atol=1e-12)
            # Independent inverse homogeneous matrix, rather than same helper.
            corrected = corrupt.copy(); corrected[:3, :3] = corrupt[:3, :3]@exp_rotation(-omega)
            expected = (np.linalg.inv(corrected)@np.column_stack([points, np.ones(2)]).T).T[:, :3]
            np.testing.assert_allclose(expected, camera_points(points, corrupt, -omega), atol=1e-12)
        self.assertAlmostEqual(rotation_error(q), np.linalg.norm(omega))

    def test_quaternion_and_visibility(self):
        m = rigid({'rotation': [np.sqrt(.5), 0, 0, np.sqrt(.5)], 'translation': [1, 2, 3]})
        np.testing.assert_allclose(m[:3, :3]@[1, 0, 0], [0, 1, 0], atol=1e-12)
        uv, good = project(np.array([[0., 0, 1], [0, 0, -1], [2, 2, 1]]), np.eye(3), 2, 2)
        self.assertEqual(good.tolist(), [True, False, False])

    def test_common_random_vectors(self):
        np.testing.assert_allclose(perturbation('scene', 1, 2), 4*perturbation('scene', 1, .5))
        self.assertFalse(np.array_equal(perturbation('scene', 0), perturbation('scene', 1)))


class Features(unittest.TestCase):
    @staticmethod
    def scan(radial, intensity=None, phase=0):
        az = np.linspace(-np.pi, np.pi, len(radial), endpoint=False)+phase
        return np.column_stack([radial*np.cos(az), radial*np.sin(az), np.ones(len(az)),
                                np.ones(len(az))*10 if intensity is None else intensity,
                                np.zeros(len(az))])

    def test_range_corner_nearer_surface_and_permutation(self):
        radial = np.r_[np.ones(128)*10, np.ones(128)*20]
        points = self.scan(radial)
        corners, ids, stats = extract_corners(points)
        self.assertGreater(stats['range'], 0)
        self.assertTrue(np.all(np.linalg.norm(corners[:, :2], axis=1) < 11))
        shuffled = np.random.default_rng(1).permutation(points)
        other, _, _ = extract_corners(shuffled)
        np.testing.assert_allclose(np.sort(corners, axis=0), np.sort(other, axis=0))
        # Changing ring ID cannot create neighbors in another scanline.
        doubled = np.vstack([points, points.copy()]); doubled[len(points):, 4] = 1
        both, _, _ = extract_corners(doubled)
        self.assertEqual(len(both), 2*len(corners))

    def test_reflectance_gap_and_no_seam_artifact(self):
        flat = self.scan(np.ones(256)*10)
        self.assertEqual(len(extract_corners(flat)[0]), 0)
        reflectance = flat.copy(); reflectance[100:160, 3] = 200
        self.assertGreater(extract_corners(reflectance)[2]['intensity'], 0)
        gap = np.delete(flat, np.arange(100, 110), axis=0)
        self.assertGreaterEqual(extract_corners(gap)[2]['gap'], 2)

    def test_quality_42_and_flat_labels(self):
        edges = np.zeros((60, 60), bool); edges[20:50, 20] = True
        corners = np.array([[-1., 0, 10], [1., 0, 10]])
        k = np.array([[20., 0, 30], [0, 20, 30], [0, 0, 1]])
        x = quality_features(corners, np.eye(4), k, edges)
        self.assertEqual(x.shape, (42,)); self.assertEqual(sum(x[:30]), 2)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(automatic_label(lambda t: 0.)[0], 0)
        self.assertEqual(automatic_label(lambda t: np.sum(t**2))[0], 1)
        self.assertEqual(automatic_label(lambda t: np.sum((t-.03)**2))[0], 0)


class Learning(unittest.TestCase):
    def test_logitboost_and_scene_leakage_guard(self):
        x = np.zeros((20, 42)); x[:, 0] = np.arange(20); y = (x[:, 0] > 9).astype(int)
        model = fit(x, y, 10)
        self.assertTrue(np.array_equal(scores(model, x) >= 0, y))
        with self.assertRaises(ValueError):
            fit(x, np.zeros(20))
        rows = [{'scene': str(i//2), 'features': [i%2]*42, 'label': i%2} for i in range(12)]
        with self.assertRaises(ValueError):
            train_scene_disjoint(rows, rows, ['heldout'])
        val = [{'scene': 'val', 'features': [i]*42, 'label': i} for i in [0, 1]]
        trained = train_scene_disjoint(rows, val, ['heldout'], rounds=5)
        self.assertEqual(len(trained['scene_cross_validation']), 5)
        self.assertNotIn('heldout', trained['training_scenes'])

    def test_burn_in_rejection_and_descent(self):
        tracker = PaperTracker(); objective = lambda t: np.sum((t-.01)**2)
        for i in range(10):
            self.assertEqual(tracker.update(objective)['status'], 'BURN_IN')
        np.testing.assert_array_equal(tracker.theta, np.zeros(3))
        # Skipped frame leaves every field bitwise unchanged, including burn-in.
        before = copy.deepcopy(tracker.__dict__)
        accepted = False
        if accepted:
            tracker.update(objective)
        for key, value in before.items():
            np.testing.assert_array_equal(value, tracker.__dict__[key])
        initial = objective(tracker.theta)
        for i in range(15):
            update = tracker.update(objective)
            self.assertLessEqual(np.max(np.abs(update['delta'])), .0024+1e-12)
        self.assertLess(objective(tracker.theta), initial)
        self.assertEqual(tracker.updates, 15)


if __name__ == '__main__':
    unittest.main()
