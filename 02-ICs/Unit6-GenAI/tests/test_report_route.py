import unittest
from app import app

class TestReportRoute(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_report_route_renders(self):
        resp = self.client.get('/report')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_data(as_text=True)
        # Ensure key sections from the template are present
        self.assertIn('Model\'s Output', data)
        self.assertIn('Key Metrics', data)

if __name__ == '__main__':
    unittest.main()

