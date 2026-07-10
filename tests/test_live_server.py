import asyncio
import unittest

from fastapi.testclient import TestClient

from step4_live_server import app


class LiveServerTests(unittest.TestCase):
    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_websocket_broadcasts_messages_to_other_clients(self):
        with TestClient(app) as client:
            with client.websocket_connect('/ws') as browser_ws:
                with client.websocket_connect('/ws') as recorder_ws:
                    recorder_ws.send_text('{"wrist": {"x": 0.1, "y": 0.2, "z": 0.3}}')
                    message = browser_ws.receive_text()
                    self.assertIn('wrist', message)


if __name__ == '__main__':
    unittest.main()
