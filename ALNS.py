import pandas as pd
import numpy as np
import math
import random
import copy
import time

# ============================================================
# 1) DATA LOADING
# ============================================================

def _to_int_list(values):
    out = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            out.append(int(v))
        except:
            s = str(v).strip()
            if s.replace(".", "", 1).isdigit():
                out.append(int(float(s)))
            else:
                out.append(s)
    return out


def read_parameters(excel_path, sheet="other parameters"):
    df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    params = {}
    for r in range(df.shape[0]):
        label = str(df.iloc[r, 0]).strip()
        if df.shape[1] > 1 and pd.notna(df.iloc[r, 1]):
            params[label] = df.iloc[r, 1]

    cap_row = df[df[0].astype(str).str.strip().eq("Capacity of compartments")]
    if not cap_row.empty:
        params["Capacity of compartments (list)"] = [
            float(v) for v in df.iloc[cap_row.index[0], 1:] if pd.notna(v)
        ]
    return params


def read_matrix_with_headers(excel_path, sheet):
    raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    col_ids = _to_int_list(raw.loc[0, 1:].tolist())
    row_ids = _to_int_list(raw.loc[1:, 0].tolist())
    body = raw.loc[1:len(row_ids), 1:len(col_ids)].apply(pd.to_numeric, errors="coerce")
    body.columns, body.index = col_ids, row_ids
    return row_ids, col_ids, body


def build_instance(excel_path):
    params = read_parameters(excel_path)
    customers, products, demand = read_matrix_with_headers(excel_path, "demand")
    nodes, _, distance = read_matrix_with_headers(excel_path, "distance")

    speed = float(params.get("Average speed (km/h)", 50.0))
    service_time = float(params.get("Service time (heure)", 0.0))

    node_list = list(nodes)  # includes depot 0
    idx = {node_id: j for j, node_id in enumerate(node_list)}

    dist_np = distance.to_numpy(dtype=float)
    tmat_np = (distance / speed).to_numpy(dtype=float)

    C = [i for i in customers if i != 0]
    P = list(products)

    return {
        "params": params,
        "V": node_list,
        "idx": idx,
        "C": C,
        "P": P,
        "demand_df": demand,
        "distance_np": dist_np,
        "travel_time_np": tmat_np,
        "service_time": service_time,
    }


def build_fleet(params):
    n_lt = int(params["Number of long-term vehicles"])
    n_st = int(params["Number of short-term vehicles"])
    caps = list(params["Capacity of compartments (list)"])

    fleet = []
    for k in range(n_lt):
        fleet.append({
            "name": f"LT{k+1}",
            "F": float(params["Fixed cost for long-term vehicle"]),
            "c": float(params["Unit cost for long-term vehicle"]),
            "Q": caps,
        })
    for k in range(n_st):
        fleet.append({
            "name": f"ST{k+1}",
            "F": float(params["Hiring cost for short-term vehicle"]),
            "c": float(params["Unit cost for short-term vehicle"]),
            "Q": caps,
        })
    return fleet


# ============================================================
# 2) CORE LOGIC
# ============================================================

class Solution:
    def __init__(self, K):
        self.routes = [[] for _ in range(K)]
        self.deliv = [dict() for _ in range(K)]
        self.rdist = [0.0 for _ in range(K)]
        self.rtime = [0.0 for _ in range(K)]

    def clone(self):
        return copy.deepcopy(self)


def _route_distance_np(route, dist, idx):
    if not route:
        return 0.0
    s = dist[idx[0], idx[route[0]]] + dist[idx[route[-1]], idx[0]]
    for a, b in zip(route[:-1], route[1:]):
        s += dist[idx[a], idx[b]]
    return float(s)


def _route_time_np(route, tmat, idx, s_time):
    if not route:
        return 0.0
    s = tmat[idx[0], idx[route[0]]] + tmat[idx[route[-1]], idx[0]]
    for a, b in zip(route[:-1], route[1:]):
        s += tmat[idx[a], idx[b]]
    s += s_time * len(route)
    return float(s)


def _recompute_vehicle_metrics(sol, inst, k):
    dist = inst["distance_np"]
    tmat = inst["travel_time_np"]
    idx = inst["idx"]
    s = inst["service_time"]
    sol.rdist[k] = _route_distance_np(sol.routes[k], dist, idx)
    sol.rtime[k] = _route_time_np(sol.routes[k], tmat, idx, s)


def _recompute_all_metrics(sol, inst):
    for k in range(len(sol.routes)):
        _recompute_vehicle_metrics(sol, inst, k)


def _vehicle_load_by_product(sol, k, P):
    load = {p: 0.0 for p in P}
    mpk = sol.deliv[k]
    for _, mp in mpk.items():
        for p, q in mp.items():
            load[p] += q
    return load


def _solution_cost(sol, fleet):
    total = 0.0
    for k, r in enumerate(sol.routes):
        if r:
            total += fleet[k]["F"] + fleet[k]["c"] * sol.rdist[k]
    return float(total)


def _best_insertion_position_np(route, i, dist, idx):
    ii = idx[i]
    if not route:
        return 0, float(dist[idx[0], ii] + dist[ii, idx[0]])

    best_pos = 0
    first = idx[route[0]]
    best_delta = float(dist[idx[0], ii] + dist[ii, first] - dist[idx[0], first])

    for pos in range(1, len(route)):
        a = idx[route[pos - 1]]
        b = idx[route[pos]]
        delta = dist[a, ii] + dist[ii, b] - dist[a, b]
        if delta < best_delta:
            best_pos, best_delta = pos, float(delta)

    last = idx[route[-1]]
    delta_last = dist[last, ii] + dist[ii, idx[0]] - dist[last, idx[0]]
    if delta_last < best_delta:
        return len(route), float(delta_last)

    return best_pos, float(best_delta)


def _delta_time_for_insertion(route, i, pos, tmat, idx, service_time):
    # delta travel time + service time for 1 extra stop
    if not route:
        return float(tmat[idx[0], idx[i]] + tmat[idx[i], idx[0]] + service_time)

    if pos == 0:
        b = route[0]
        return float(tmat[idx[0], idx[i]] + tmat[idx[i], idx[b]] - tmat[idx[0], idx[b]] + service_time)

    if pos == len(route):
        a = route[-1]
        return float(tmat[idx[a], idx[i]] + tmat[idx[i], idx[0]] - tmat[idx[a], idx[0]] + service_time)

    a = route[pos - 1]
    b = route[pos]
    return float(tmat[idx[a], idx[i]] + tmat[idx[i], idx[b]] - tmat[idx[a], idx[b]] + service_time)


# ============================================================
# 3) DESTROY & REPAIR
# ============================================================

def _destroy_any_fast(sol, inst, fleet, mode="shaw", remove_frac=0.30, rng=None):
    rng = rng or random
    dist = inst["distance_np"]
    idx = inst["idx"]

    visits = [(k, i) for k, r in enumerate(sol.routes) for i in r]
    if not visits:
        return {i: {p: 0.0 for p in inst["P"]} for i in inst["C"]}

    if mode == "shaw":
        _, pivot = rng.choice(visits)
        ip = idx[pivot]
        visits.sort(key=lambda x: dist[ip, idx[x[1]]])

    elif mode == "worst":
        worst_list = []
        for k, i in visits:
            r = sol.routes[k]
            pos = r.index(i)
            prev_node = 0 if pos == 0 else r[pos - 1]
            next_node = 0 if pos == len(r) - 1 else r[pos + 1]
            saving = dist[idx[prev_node], idx[i]] + dist[idx[i], idx[next_node]] - dist[idx[prev_node], idx[next_node]]
            worst_list.append((saving * fleet[k]["c"], k, i))
        worst_list.sort(key=lambda x: x[0], reverse=True)
        visits = [(k, i) for _, k, i in worst_list]

    else:
        rng.shuffle(visits)

    m = max(1, int(remove_frac * len(visits)))
    chosen = visits[:m]

    unserved = {i: {p: 0.0 for p in inst["P"]} for i in inst["C"]}
    to_remove = {}
    for k, i in chosen:
        to_remove.setdefault(k, set()).add(i)

    # Recover deliveries
    for k, rm_nodes in to_remove.items():
        for i in list(rm_nodes):
            if i in sol.deliv[k]:
                for p, q in sol.deliv[k][i].items():
                    unserved[i][p] += q
                del sol.deliv[k][i]

    # Remove nodes (batch) + refresh metrics
    for k, rm_nodes in to_remove.items():
        sol.routes[k] = [x for x in sol.routes[k] if x not in rm_nodes]
        _recompute_vehicle_metrics(sol, inst, k)

    return unserved


def _repair_split_fast(sol, inst, fleet, unserved, mode="regret", topK_vehicles=None):
    C, P = inst["C"], inst["P"]
    dist = inst["distance_np"]
    tmat = inst["travel_time_np"]
    idx = inst["idx"]
    s = inst["service_time"]

    Dkmax = float(inst["params"]["Maximum distance (km)"])
    Tkmax = float(inst["params"]["Maximum time (heure)"])

    K_all = list(range(len(fleet)))

    # cache loads per vehicle
    loads = []
    for k in K_all:
        loads.append(_vehicle_load_by_product(sol, k, P))

    while True:
        active_clients = [c for c in C if sum(unserved[c].values()) > 1e-6]
        if not active_clients:
            break

        candidates = []

        for i in active_clients:
            un_i = unserved[i]

            # choose vehicles to test
            if topK_vehicles is not None and topK_vehicles < len(K_all):
                ii = idx[i]
                base = dist[idx[0], ii] + dist[ii, idx[0]]
                K_list = sorted(K_all, key=lambda k: fleet[k]["c"] * base)[:topK_vehicles]
            else:
                K_list = K_all

            veh_costs = []

            for k in K_list:
                load_k = loads[k]
                Qk = fleet[k]["Q"]

                # compute how much can be delivered (same logic)
                can_deliver = 0.0
                for j, p in enumerate(P):
                    cap_left = Qk[j] - load_k[p]
                    if cap_left > 0.0:
                        qtake = un_i[p]
                        if qtake > 0.0:
                            can_deliver += (qtake if qtake < cap_left else cap_left)

                if can_deliver <= 1e-9:
                    continue

                pos, delta_dist = _best_insertion_position_np(sol.routes[k], i, dist, idx)
                delta_time = _delta_time_for_insertion(sol.routes[k], i, pos, tmat, idx, s)

                # feasibility via caches
                if sol.rdist[k] + delta_dist > Dkmax + 1e-6:
                    continue
                if sol.rtime[k] + delta_time > Tkmax + 1e-6:
                    continue

                veh_costs.append((delta_dist / can_deliver, k, pos, can_deliver, delta_dist, delta_time))

            if not veh_costs:
                continue

            veh_costs.sort(key=lambda x: x[0])
            best = veh_costs[0]

            if mode == "regret":
                regret = (veh_costs[1][0] - veh_costs[0][0]) if len(veh_costs) > 1 else 2000.0
                candidates.append((regret, i, best[1], best[2], best[4], best[5]))
            else:
                score = best[3] / (1.0 + abs(best[0]))
                candidates.append((score, i, best[1], best[2], best[4], best[5]))

        if not candidates:
            return False

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, bi, bk, bp, dd, dt = candidates[0]

        if bi not in sol.routes[bk]:
            sol.routes[bk].insert(bp, bi)
            sol.rdist[bk] += dd
            sol.rtime[bk] += dt

        # update deliveries + loads
        load_k = loads[bk]
        Qk = fleet[bk]["Q"]
        un_bi = unserved[bi]

        for j, p in enumerate(P):
            cap_left = Qk[j] - load_k[p]
            if cap_left <= 0.0:
                continue
            qty = un_bi[p]
            if qty <= 0.0:
                continue

            take = qty if qty < cap_left else cap_left
            if take > 1e-9:
                sol.deliv[bk].setdefault(bi, {})
                sol.deliv[bk][bi][p] = sol.deliv[bk][bi].get(p, 0.0) + take
                un_bi[p] -= take
                load_k[p] += take

    return True


def _repair_with_fallback(sol, inst, fleet, unserved, mode, topK_vehicles):
    """
    We tried fast repair with topK. When it fails, we retry once with all vehicles.
    This prevents "no solution" on larger instances.
    """
    ok = _repair_split_fast(sol, inst, fleet, unserved, mode=mode, topK_vehicles=topK_vehicles)
    if ok:
        return True
    # fallback
    return _repair_split_fast(sol, inst, fleet, unserved, mode=mode, topK_vehicles=None)


# ============================================================
# 4) MAIN SOLVER
# ============================================================

def solve_mc_vrp_alns(
    excel_file,
    seed=42,
    iters=10_000_000,
    time_limit_sec=60,     # hard time cap
    remove_frac=0.30,
    alpha=0.9992,
    topK_vehicles=8,       # used only in ALNS iterations (not in initial build)
    print_every=500
):
    inst = build_instance(excel_file)
    fleet = build_fleet(inst["params"])
    rng = random.Random(seed)
    t_start = time.time()

    # --------------------------
    # INITIAL SOLUTION
    # Build with ALL vehicles (no topK) to avoid infeasible starts
    # --------------------------
    cur = Solution(len(fleet))
    unserved0 = {i: {p: float(inst["demand_df"].loc[i, p]) for p in inst["P"]} for i in inst["C"]}

    ok_init = _repair_split_fast(cur, inst, fleet, unserved0, mode="regret", topK_vehicles=None)
    if not ok_init:
        return None  # cannot even build a feasible solution

    _recompute_all_metrics(cur, inst)

    best = cur.clone()
    cur_cost = best_cost = _solution_cost(cur, fleet)

    # SA temp
    T = max(1.0, 0.05 * cur_cost)
    last_imp = 0

    for it in range(1, iters + 1):
        if (time.time() - t_start) >= time_limit_sec:
            break

        if it - last_imp > 600:
            T = best_cost * 0.04
            last_imp = it

        cand = cur.clone()

        rnd = rng.random()
        d_mode = "shaw" if rnd < 0.4 else ("worst" if rnd < 0.7 else "random")
        unserved = _destroy_any_fast(cand, inst, fleet, mode=d_mode, remove_frac=remove_frac, rng=rng)

        r_mode = "regret" if rng.random() < 0.5 else "greedy"

        # Repair with fallback to avoid "no solution"
        if _repair_with_fallback(cand, inst, fleet, unserved, mode=r_mode, topK_vehicles=topK_vehicles):
            c_cost = _solution_cost(cand, fleet)

            # SA acceptance
            if c_cost < cur_cost - 1e-7 or rng.random() < math.exp((cur_cost - c_cost) / max(1e-9, T)):
                cur, cur_cost = cand, c_cost
                if c_cost < best_cost - 1e-7:
                    best, best_cost, last_imp = cand.clone(), c_cost, it

        T *= alpha

        if print_every and (it % print_every == 0):
            print(f"It {it:6d} | Best {best_cost:10.2f} | Cur {cur_cost:10.2f} | T {T:8.2f} | Time {time.time()-t_start:6.1f}s")

    return {
        "obj": float(best_cost),
        "time": float(time.time() - t_start),
        "routes": [
            {"veh": f["name"], "path": [0] + best.routes[k] + [0]}
            for k, f in enumerate(fleet)
            if best.routes[k]
        ],
    }


# ============================================================
# 5) RUN
# ============================================================

if __name__ == "__main__":
    file_path = "/Users/theachaduteau/Documents/GurobiPython/Data/portugal_100.xlsx"
    

    res = solve_mc_vrp_alns(
        file_path,
        seed=42,
        time_limit_sec=120,   # time limit hardcap
        remove_frac=0.30,
        topK_vehicles=6,     # speed knob; fallback prevents infeasibility
        print_every=500
    )

    if not res:
        print("\nAucune solution faisable trouvée (même l'initialisation a échoué).")
    else:
        print(f"\nObjectif: {res['obj']}")
        print(f"Temps de résolution : {res['time']:.2f} s")
        for r in res["routes"]:
            print(f"{r['veh']}: {r['path']}")
