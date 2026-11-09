"""Accounting note for a discarded numerical attempt; not a result estimate."""
from ab_common import save

if __name__=='__main__':
    save('ab_run_audit.json',dict(classification='P',source_script='ab_run_audit.py',
        failed_run_last_observed_process_CPU_s=924.3125,
        failed_run_conservative_budget_charge_s=1200.,auxiliary_analysis_budget_allowance_s=300.,
        note='The last process CPU observation preceded failure; 1200 s is an accounting allowance, not a measured total. Auxiliary scripts/analysis are allowed another 300 s. The failed run was rejected for a nonreal Hermitian LU pivot, and supplied no large-cell result.'))
