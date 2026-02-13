import unittest
import os

class TestSummaryTemplate(unittest.TestCase):
    def setUp(self):
        self.template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'model_report.html')
        self.template_path = os.path.normpath(self.template_path)

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.template_path), f"Template not found at {self.template_path}")

    def test_contains_sections(self):
        with open(self.template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key placeholders and structure
        self.assertIn('Model\'s Output', content)
        # self.assertIn('Key Findings', content)
        self.assertIn('Recommendations', content)
        self.assertIn('Key Metrics', content)
        self.assertIn('metricsChart', content)  # canvas id
        self.assertIn('{{ report_title', content) or 'Report Summary' in content

if __name__ == '__main__':
    unittest.main()

