"""실제 키나 API 호출 없이 Cloud와 로컬의 키 설정을 검사합니다."""

import unittest
from unittest.mock import patch

from app import read_api_key


class SecretTests(unittest.TestCase):
    def test_cloud_secrets_take_priority(self):
        with patch("app.st.secrets", {"OPENAI_API_KEY": " cloud-test-value "}), \
                patch("app.dotenv_values") as dotenv:
            self.assertEqual(read_api_key(), "cloud-test-value")
            dotenv.assert_not_called()

    def test_local_without_secrets_file(self):
        with patch("app.st.secrets") as secrets, \
                patch("app.dotenv_values", return_value={"OPENAI_API_KEY": "local-test-value"}):
            secrets.get.side_effect = FileNotFoundError("No secrets file")
            self.assertEqual(read_api_key(), "local-test-value")

    def test_missing_secrets_key_falls_back(self):
        with patch("app.st.secrets", {}), \
                patch("app.dotenv_values", return_value={"OPENAI_API_KEY": "local-test-value"}):
            self.assertEqual(read_api_key(), "local-test-value")

    def test_blank_secret_falls_back(self):
        with patch("app.st.secrets", {"OPENAI_API_KEY": "  "}), \
                patch("app.dotenv_values", return_value={"OPENAI_API_KEY": "local-test-value"}):
            self.assertEqual(read_api_key(), "local-test-value")

    def test_no_key_returns_empty(self):
        with patch("app.st.secrets", {}), patch("app.dotenv_values", return_value={}):
            self.assertEqual(read_api_key(), "")

    def test_invalid_secret_type(self):
        with patch("app.st.secrets", {"OPENAI_API_KEY": 1234}):
            with self.assertRaises(ValueError):
                read_api_key()


if __name__ == "__main__":
    unittest.main()
