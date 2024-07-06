from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import CIT
import multiprocessing as mp

from itertools import combinations, permutations
import time
import ray

def skeleton_discovery(
    data: ndarray, 
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    print("startPC_skeleton")
    #global pool
    #pool = mp.pool.ThreadPool(64)
    tbefore = time.time()
    '''
    ray.init(
        num_cpus = 4,
        _plasma_directory="/tmp",_system_config={
        #RAY_lineage_pinning_enabled=0
        "local_fs_capacity_threshold": 1,
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps(
            {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        )
    }#,
        #runtime_env= {"RAY_memory_monitor_refresh_ms": 0}
            )
    '''
    tafter = time.time()
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    rough_skeleton = np.zeros((no_of_var, no_of_var)).astype(int)
    
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    node_ids = list(range(no_of_var))
    cg.depth_time = []
    cg.independence_tests = 0
    while cg.max_degree()-1 > depth:
        #tstart = time.time()
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for (x, y) in permutations(node_ids, 2):
            if show_progress:
                pbar.update()
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) <= depth - 1:
                continue
            if y not in Neigh_x:
                continue
            #skip when degree of y is less than required degree
            '''
            if len(cg.neighbors(y)) < depth - 1:
                continue
            '''
            knowledge_ban_edge = False
            sepsets = set()
            if background_knowledge is not None and (
                    background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                knowledge_ban_edge = True
            if knowledge_ban_edge:
                if not stable:
                    edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                    if edge1 is not None:
                        cg.G.remove_edge(edge1)
                    edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                    if edge2 is not None:
                        cg.G.remove_edge(edge2)
                    append_value(cg.sepset, x, y, ())
                    append_value(cg.sepset, y, x, ())
                else:
                    edge_removal.append((x, y))  # after all conditioning sets at
                    edge_removal.append((y, x))  # depth l have been considered
            Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
            S_candidate = []
            for S in combinations(Neigh_x_noy, depth):
                #case1 check
                
                s_bad = False
                for s in S:
                    if rough_skeleton[s,y] == 1 and rough_skeleton[y,s] == 1:
                        s_bad = True
                        break
                if s_bad == True:
                    continue
                
                #case2 delay
                '''
                s_bad = False
                if len(S) >1:
                    for si1 in range(len(S)-1):
                        if s_bad == True:
                            break
                        s1 = S[si1]
                        for si2 in range(si1+1, len(S)):
                            if s_bad == True:
                                break
                            s2 = S[si2]
                            if cg.sepset[s1][s2] is None or rough_skeleton[s1,s2] == 1:
                                continue
                            s_bad = True
                            for seps in cg.sepset[s1][s2]:
                                for sep in seps:
                                    if sep == x:
                                        s_bad = False
                if s_bad == True:
                    S_candidate.append(S)
                    continue
                '''
                #start independence test
                p = cg.ci_test(x, y, S)
                cg.independence_tests+=1
                if p > alpha:
                    if verbose:
                        print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, S)
                        append_value(cg.sepset, y, x, S)
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered
                        for s in S:
                            sepsets.add(s)
                else:
                    if verbose:
                        print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
            #check case2 candidates
            '''
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if (stable and (x,y) not in edge_removal) or (not stable and edge1 is not None):
                for S in S_candidate:
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
            '''
            append_value(cg.sepset, x, y, tuple(sepsets))
            append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()
        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)
                        
        if depth==0:
            for (x, y) in permutations(node_ids, 2):
                if cg.G.get_edge(cg.G.nodes[x],cg.G.nodes[y]) is None:
                    rough_skeleton[x,y] = 1
        print("independence tests:", cg.independence_tests)
    cg.depth = depth
    cg.ratio = depth/cg.max_degree()
    return cg