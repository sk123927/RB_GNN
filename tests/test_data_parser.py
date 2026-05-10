from pathlib import Path
import tempfile
import unittest

from rbgnn.data import read_diffusion_graph, read_pair_file, validate_samples


class DataParserTest(unittest.TestCase):
    def test_pair_file_uses_four_non_empty_records_per_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.txt"
            path.write_text(
                "0 2\n"
                "1 3\n"
                "5.0\n"
                "2.0\n"
                "\n"
                "1\n"
                "0\n"
                "4.0\n"
                "1.0\n"
            )
            samples = read_pair_file(path)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].rumor_seeds, [0, 2])
        self.assertEqual(samples[0].protectors, [1, 3])
        self.assertEqual(samples[1].budget, 1)

    def test_diffusion_graph_preserves_edge_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.txt"
            path.write_text("0 1 10 4 0.2\n1 2 3 2 0.4\n")
            graph = read_diffusion_graph(path, num_nodes=4)

        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(graph.attribute_names[-1], "propagation_probability")
        self.assertEqual(graph.out_degrees(), [1, 1, 0, 0])
        self.assertEqual(graph.in_degrees(), [0, 1, 1, 0])

    def test_validate_samples_rejects_out_of_range_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.txt"
            path.write_text("0 5\n2\n3.0\n1.0\n")
            samples = read_pair_file(path)
            with self.assertRaises(ValueError):
                validate_samples(samples, num_nodes=4)


if __name__ == "__main__":
    unittest.main()
