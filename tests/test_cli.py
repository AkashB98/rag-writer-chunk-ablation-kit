import sys, os, json, subprocess, threading, unittest, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(os.path.dirname(__file__), "..")


def run_cli(*argv):
    return subprocess.run([sys.executable, "cli.py", *argv], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


class TestCLI(unittest.TestCase):
    def test_check_exit_0(self):
        r = run_cli("check")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_check_bad_regime_exit_2(self):
        r = run_cli("check", "--regime", "Z")
        self.assertEqual(r.returncode, 2)

    def test_check_bad_n_exit_2(self):
        r = run_cli("check", "--n", "0")
        self.assertEqual(r.returncode, 2)

    def test_run_bad_regime_exit_2(self):
        r = run_cli("run", "--regime", "Z")
        self.assertEqual(r.returncode, 2)

    def test_demo_exit_0(self):
        r = run_cli("demo")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("reversal holds: True", r.stdout)

    def test_ablate_json_shape(self):
        r = run_cli("ablate")
        self.assertEqual(r.returncode, 0)
        data = json.loads(r.stdout)
        self.assertEqual(len(data["cells"]), 24)
        self.assertTrue(data["reversal"]["holds"])

    def test_generate_to_file(self):
        out = "/tmp/rag_ablation_corpus_test.json"
        r = run_cli("generate", "--output", out)
        self.assertEqual(r.returncode, 0)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(len(data["documents"]), 20)
        self.assertEqual(len(data["queries"]), 20)

    def test_report_contains_table(self):
        r = run_cli("report")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Factor deltas", r.stdout)
        self.assertIn("expanded_tfidf", r.stdout)


class TestServe(unittest.TestCase):
    def test_health_and_ablate(self):
        from http.server import HTTPServer
        import cli as cli_mod
        server = HTTPServer(("127.0.0.1", 0), cli_mod._Handler)
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
                self.assertEqual(json.loads(resp.read())["status"], "ok")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/ablate",
                data=json.dumps({"n": 4}).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            self.assertEqual(len(data["cells"]), 24)
            self.assertTrue(data["reversal"]["holds"])
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
