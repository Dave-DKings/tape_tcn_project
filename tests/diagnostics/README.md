# 🔍 Diagnostic Scripts

This folder contains diagnostic scripts that were used to identify and verify fixes for specific issues during development.

---

## 📁 Files (9 scripts)

### **Asset Memory Diagnostics**
- **`test_asset_memory_approach.py`** - Test asset memory management approach
- **`test_asset_memory_fix.py`** - Verify fix for asset memory issue

### **General Fixes**
- **`test_corrected_approach.py`** - Test corrected implementation approach
- **`test_final_fix.py`** - Verify final fix for resolved issue

### **TCN Diagnostics**
- **`test_tcn_agent_fix.py`** - Test TCN agent bug fix

### **Training Diagnostics**
- **`test_num_updates_issue.py`** - Test fix for num_updates calculation

### **Feature/Evaluation Diagnostics**
- **`diagnose_feature_transformation.py`** - Diagnose feature transformation pipeline
- **`verify_eigenvalue_exclusion.py`** - Verify eigenvalue columns properly excluded
- **`verify_tcn_stochastic.py`** - Verify TCN stochastic behavior

---

## 🎯 Purpose

These scripts were created to:
1. Diagnose specific bugs and issues
2. Verify that fixes were correctly applied
3. Test edge cases and corner scenarios
4. Validate data transformations

---

## 📝 Usage

Most of these scripts are **one-time diagnostics** that have already served their purpose. They are kept for:
- Historical reference
- Re-verification if needed
- Understanding the debugging process

To run any diagnostic:
```bash
cd /workspaces/multi_modal_drl_continuous/adaptive_portfolio_rl
python tests/diagnostics/<script_name>.py
```

---

## ⚠️ Note

These scripts may depend on specific data states or configurations that existed when they were created. They might need updates to work with the current codebase.

For **current validation**, use the scripts in the root directory:
- `verify_fixes.py` - Comprehensive fix verification
- `diagnostic_scaling_audit.py` - Data scaling audit
- `test_reproducibility.py` - Reproducibility testing

---

**Last Updated:** October 19, 2025  
**Status:** Archived for reference
