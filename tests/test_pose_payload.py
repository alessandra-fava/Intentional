import unittest
import numpy as np

from step3_record_reachability import build_pose_payload


class PosePayloadTests(unittest.TestCase):
    def test_build_pose_payload_contains_smoothed_wrist(self):
        payload = build_pose_payload(1.23, np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), valid=True)
        self.assertEqual(payload['timestamp'], 1.23)
        self.assertTrue(payload['valid'])
        self.assertAlmostEqual(payload['wrist']['x'], 0.4)
        self.assertAlmostEqual(payload['wrist']['y'], 0.5)
        self.assertAlmostEqual(payload['wrist']['z'], 0.6)

    def test_build_pose_payload_includes_joint_positions(self):
        joints = {
            'right_shoulder': np.array([0.1, 0.0, 0.2]),
            'right_elbow': np.array([0.2, 0.0, 0.1]),
            'right_wrist': np.array([0.3, 0.0, 0.0]),
        }
        payload = build_pose_payload(2.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), joints=joints, valid=True)
        self.assertIn('joints', payload)
        self.assertIn('right_wrist', payload['joints'])
        self.assertAlmostEqual(payload['joints']['right_wrist']['x'], 0.3)


if __name__ == '__main__':
    unittest.main()
