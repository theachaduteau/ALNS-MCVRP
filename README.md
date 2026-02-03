# MC-VRP ALNS Solver (Split Deliveries, LT/ST Fleet)

Python implementation of an **Adaptive Large Neighborhood Search (ALNS)** with **Simulated Annealing** acceptance for an industrial-style **Multi-Compartment VRP** with:
- **Multiple products** (each mapped to a compartment)
- **Capacity constraints per compartment**
- **Split deliveries** (a customer’s demand can be delivered by multiple vehicles / visits across the fleet)
- **Fleet mix**: long-term (LT) and short-term (ST) vehicles with distinct fixed + variable costs
- Route feasibility constraints:
  - Maximum route **distance** (km)
  - Maximum route **time** (hours), including service time per stop

The code loads an instance from an Excel file, builds an initial feasible solution, then iteratively improves it using destroy/repair operators.



## 1) Repository / File structure

This project is currently a single Python script containing:

- **Data loading**
  - Read parameters + matrices from Excel
- **Core solution structure**
  - Routes, deliveries, cached route distance/time
- **Destroy & repair**
  - Shaw / Worst / Random destroy
  - Greedy / Regret repair with fallback
- **Main solver**
  - ALNS loop + Simulated Annealing acceptance
- **CLI entry point**
  - Example run in `if __name__ == "__main__":`



## 2) Requirements

### Python version
- Python **3.9+** recommended

### Dependencies
```bash
pip install pandas numpy openpyxl

