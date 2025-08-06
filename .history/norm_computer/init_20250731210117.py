diff --git a//dev/null b/norm_computer/__init__.py
index 0000000000000000000000000000000000000000..891c5a5d0bab150368822337fb518fd373078cf4 100644
--- a//dev/null
+++ b/norm_computer/__init__.py
@@ -0,0 +1,33 @@
+"""Normalization utilities subpackage
+
+Provides helper functions to compute tables used to normalize lensing statistics.
+The most common routines from :mod:`compute_norm_grid` and
+:mod:`compute_norm_grid_base` are re-exported here for convenience."""
+
+from .compute_norm_grid import (
+    logRe_of_logMsps,
+    generate_lens_samples_no_alpha as generate_samples_4d,
+    compute_A_phys_eta as compute_A_phys_eta_4d,
+    single_A_eta_entry as single_entry_4d,
+    build_A_phys_table_parallel_4D,
+)
+from .compute_norm_grid_base import (
+    generate_lens_samples_no_alpha,
+    compute_A_phys_eta,
+    single_A_eta_entry,
+    build_A_phys_table_parallel,
+)
+
+__all__ = [
+    "logRe_of_logMsps",
+    "generate_samples_4d",
+    "compute_A_phys_eta_4d",
+    "single_entry_4d",
+    "build_A_phys_table_parallel_4D",
+    "generate_lens_samples_no_alpha",
+    "compute_A_phys_eta",
+    "single_A_eta_entry",
+    "build_A_phys_table_parallel",
+]
+
+__version__ = "0.1"
