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
import json
from copy import deepcopy

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
    global pool
    pool = mp.pool.ThreadPool(4)
    tbefore = time.time()
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    rough_skeleton = np.zeros((no_of_var, no_of_var)).astype(int)
    
    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    node_ids = list(range(no_of_var))
    curr_partition = 0
    max_partition = 0
    cg.depth_time = []
    while cg.max_degree()-1 > depth:
        #tstart = time.time()
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for (x, y) in permutations(node_ids, 2):
            if show_progress:
                pbar.update()
            #if show_progress:
                #pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) <= depth - 1:
                continue
            if y not in Neigh_x:
                continue
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
        print("check partition")
        tt = time.time()
        if  cg.max_degree() - 1 > depth and partition_available(cg, node_ids):
            max_partition+=1
            print("partition time:",time.time()-tt)
            break
        print("partition time:",time.time()-tt)
    cg.depth = depth
    cg.ratio = depth/cg.max_degree()
    #return cg
    print("Rough end",depth,cg.max_degree())
    cg.partitions = 0
    
    maxDegree = cg.max_degree()
    avgDegree = np.mean(np.sum(cg.G.graph != 0, axis=1))
    cg = skeleton_partition(curr_partition, max_partition, depth, no_of_var , cg, node_ids, alpha, indep_test, stable, background_knowledge, verbose, show_progress, node_names)
    if show_progress:
        pbar.close()
    #pool.close()
    
    cg.PC_elapsed = time.time()-tbefore
    cg.maxDegree = maxDegree
    cg.avgDegree = avgDegree
    return cg

def partition_available(
    cg: CausalGraph,
    node_ids: List[int],
) -> CausalGraph:
    print("start building L")
    tt = time.time()
    edge_size = cg.G.get_num_edges()
    edge_set = set()
    edge_list = []
    edge_id = {}
    
    node_id = {}
    node_set = set()
    node_list = []
    L = np.zeros((edge_size, edge_size))
    for i in node_ids:
        node_id[i] = len(node_list)
        node_set.add(i)
        node_list.append(i)
    for i in node_ids:
        for j in node_ids:
            if node_id[i]>=node_id[j]:
                continue
            if cg.G.graph[node_id[i], node_id[j]] == 0:
                continue
            edge_id[(i,j)] = len(edge_list)
            edge_id[(j,i)] = len(edge_list)
            edge_list.append((i,j))
    for e1 in range(len(edge_list) - 1):
        edge1 = edge_list[e1]
        for e2 in range(e1+1, len(edge_list)):
            edge2 = edge_list[e2]
            if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]:
                L[e1,e2] = -1
                L[e2,e1] = -1
                L[e1,e1] += 1
                L[e2,e2] += 1
    print("Build L end: ", time.time()-tt)
    eigen_values, eigen_vectors = np.linalg.eigh(L)
    eigen_vector = (eigen_vectors[:,1] >= 0).astype(int)
    print("eigen vector ended: ", tt-time.time())
    tt = time.time()
    node_set.clear()
    edge_set.clear()
    edge_id.clear()
    nodes_A = set()
    nodes_B = set()
    for id1 in range(len(eigen_vector)):
        if eigen_vector[id1] > 0:
            nodes_A.add(edge_list[id1][0])
            nodes_A.add(edge_list[id1][1])
        else:
            nodes_B.add(edge_list[id1][0])
            nodes_B.add(edge_list[id1][1])
    
    for id1 in range(len(eigen_vector)):
        for id2 in range(id1, len(eigen_vector)):
            if id1 == id2:
                continue
            if (eigen_vector[id1] == 0 and eigen_vector[id2] == 1) and (L[id1, id2] == -1):
                nodes_B.add(edge_list[id2][0])
                nodes_B.add(edge_list[id2][1])
                nodes_A.add(edge_list[id1][0])
                nodes_A.add(edge_list[id1][1])
            if (eigen_vector[id2] == 0 and eigen_vector[id1] == 1) and (L[id1, id2] == -1):
                nodes_B.add(edge_list[id1][0])
                nodes_B.add(edge_list[id1][1])
                nodes_A.add(edge_list[id2][0])
                nodes_A.add(edge_list[id2][1])
    
    print("check ABC end: ", time.time()-tt)
    if len(nodes_A) > 0 and len(nodes_B)> 0 and (len(nodes_A) < len(node_ids)*3/4 and len(nodes_B) < len(node_ids)*3/4):
        return True
    else:
        return False


def skeleton_partition(#rough_skeleton,
    curr_partition,
    max_partition,
    depth,
    origin_size,
    cg: CausalGraph,
    node_ids: List[int],
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    global pool
    #print(curr_partition, max_partition)
    if curr_partition >= max_partition:
        return skeleton_conditional(curr_partition, max_partition, depth, origin_size, cg, node_ids, alpha, indep_test, stable, background_knowledge, verbose, show_progress, node_names)
    partitions = 1
    edge_size = cg.G.get_num_edges()
    edge_set = set()
    edge_list = []
    edge_id = {}
    
    node_id = {}
    node_set = set()
    node_list = []
    #Build matrix L
    L = np.zeros((edge_size, edge_size))
    tstart = time.time()
    for i in node_ids:
        node_id[i] = len(node_list)
        node_set.add(i)
        node_list.append(i)
        
    for i in node_ids:
        for j in node_ids:
            if node_id[i]>=node_id[j]:
                continue
            if cg.G.graph[node_id[i], node_id[j]] == 0:
                continue
            edge_id[(i,j)] = len(edge_list)
            edge_id[(j,i)] = len(edge_list)
            edge_list.append((i,j))
    for e1 in range(len(edge_list) - 1):
        edge1 = edge_list[e1]
        for e2 in range(e1+1, len(edge_list)):
            edge2 = edge_list[e2]
            if edge1[0] == edge2[0] or edge1[0] == edge2[1] or edge1[1] == edge2[0] or edge1[1] == edge2[1]:
                L[e1,e2] = -1
                L[e2,e1] = -1
                L[e1,e1] += 1
                L[e2,e2] += 1
    print("building L ended: ", tstart-time.time())
    tt = time.time()
    #Partition through eigen vector
    eigen_values, eigen_vectors = np.linalg.eigh(L)
    eigen_vector = (eigen_vectors[:,1] >= 0).astype(int)
    print("eigen vector ended: ", tt-time.time())
    tt = time.time()
    node_set.clear()
    edge_set.clear()
    edge_id.clear()
    nodes_A = set()
    nodes_B = set()
    for id1 in range(len(eigen_vector)):
        if eigen_vector[id1] > 0:
            nodes_A.add(edge_list[id1][0])
            nodes_A.add(edge_list[id1][1])
        else:
            nodes_B.add(edge_list[id1][0])
            nodes_B.add(edge_list[id1][1])
    
    for id1 in range(len(eigen_vector)):
        for id2 in range(id1, len(eigen_vector)):
            if id1 == id2:
                continue
            if (eigen_vector[id1] == 0 and eigen_vector[id2] == 1) and (L[id1, id2] == -1):
                nodes_B.add(edge_list[id2][0])
                nodes_B.add(edge_list[id2][1])
                nodes_A.add(edge_list[id1][0])
                nodes_A.add(edge_list[id1][1])
            if (eigen_vector[id2] == 0 and eigen_vector[id1] == 1) and (L[id1, id2] == -1):
                nodes_B.add(edge_list[id1][0])
                nodes_B.add(edge_list[id1][1])
                nodes_A.add(edge_list[id2][0])
                nodes_A.add(edge_list[id2][1])
    
    print("get nodes ended", time.time()-tt)
    print("total partition time: ", time.time()-tstart)
    
    if len(nodes_A) > len(nodes_B):
        tmp = nodes_A
        nodes_A = nodes_B
        nodes_B = tmp
    nodes_A = list(nodes_A)
    nodes_B = list(nodes_B)
    print("partitioned length: ", len(nodes_A), len(nodes_B))
    nodeA_id = {}
    nodeB_id = {}
    for (iA, idA) in enumerate(nodes_A):
        nodeA_id[idA] = iA
    for (iB, idB) in enumerate(nodes_B):
        nodeB_id[idB] = iB
    cgA = CausalGraph(len(nodes_A), nodes_A)
    cgB = CausalGraph(len(nodes_B), nodes_B)
    cgA.set_ind_test(indep_test)
    cgB.set_ind_test(indep_test)
    for x in nodes_A:
        for y in nodes_A:
            if (x != y) and cg.G.get_edge(cg.G.nodes[node_id[x]], cg.G.nodes[node_id[y]]) is None:
                edge1 = cgA.G.get_edge(cgA.G.nodes[nodeA_id[x]], cgA.G.nodes[nodeA_id[y]])
                if edge1 is not None:
                    cgA.G.remove_edge(edge1)
    for x in nodes_B:
        for y in nodes_B:
            if (x != y) and cg.G.get_edge(cg.G.nodes[node_id[x]], cg.G.nodes[node_id[y]]) is None:
                edge1 = cgB.G.get_edge(cgB.G.nodes[nodeB_id[x]], cgB.G.nodes[nodeB_id[y]])
                if edge1 is not None:
                    cgB.G.remove_edge(edge1)
    if node_names is not None:
        node_nameA = list(np.array(node_names)[[node_id[nA] for nA in nodes_A]])
        node_nameB = list(np.array(node_names)[[node_id[nA] for nA in nodes_B]])
    else:
        node_nameA = None
        node_nameB = None
    curr_partition += 1
    
    #multiprocessing
    st = time.time()
    print("multithreading start", mp.cpu_count() )
    pargs = []
    pargs.append((curr_partition, max_partition, depth, origin_size, cgA, nodes_A, alpha, indep_test, stable, background_knowledge, verbose, show_progress, node_nameA))
    pargs.append((curr_partition, max_partition, depth, origin_size, cgB, nodes_B, alpha, indep_test, stable, background_knowledge, verbose, show_progress, node_nameB))
    [cgA, cgB] = pool.starmap(skeleton_partition, pargs)
    pool.close()
    pool.join()
    print("multithreading end:", time.time()-st)
    cg.ts = time.time()-st
    tt = time.time()
    
    #Merge A and B edges
    nodes_A_set = set(nodes_A)
    nodes_B_set = set(nodes_B)
    for x in node_ids:
        for y in node_ids:
            if x == y:
                continue
            if (x in nodes_A_set) and (y in nodes_A_set):
                edge1 = cgA.G.get_edge(cgA.G.nodes[nodeA_id[x]], cgA.G.nodes[nodeA_id[y]])
                if edge1 is None:
                    edgeg = cg.G.get_edge(cg.G.nodes[node_id[x]], cg.G.nodes[node_id[y]])
                    if edgeg is not None:
                        cg.G.remove_edge(edgeg)
            
            if (x in nodes_B_set) and (y in nodes_B_set):
                edge1 = cgB.G.get_edge(cgB.G.nodes[nodeB_id[x]], cgB.G.nodes[nodeB_id[y]])
                if edge1 is None:
                    edgeg = cg.G.get_edge(cg.G.nodes[node_id[x]], cg.G.nodes[node_id[y]])
                    if edgeg is not None:
                        cg.G.remove_edge(edgeg)
    #Merge A and B sepset
    for idx1 in range(len(node_ids)):
        for idx2 in range(len(node_ids)):
            if cg.sepset[idx1][idx2] is not None:
                sepset_change = set(cg.sepset[idx1][idx2])
            else:
                sepset_change = set()
            if (node_ids[idx1] in nodeA_id) and (node_ids[idx2] in nodeA_id):
                sepA = cgA.sepset[nodeA_id[node_ids[idx1]]][nodeA_id[node_ids[idx2]]]
                if sepA is not None:
                    if len(sepset_change) == 0:
                        sepset_change = set(cgA.sepset[nodeA_id[node_ids[idx1]]][nodeA_id[node_ids[idx2]]])
                    else:
                        sepset_change |= set(cgA.sepset[nodeA_id[node_ids[idx1]]][nodeA_id[node_ids[idx2]]])
            
            if (node_ids[idx1] in nodeB_id) and (node_ids[idx2] in nodeB_id):
                if cgB.sepset[nodeB_id[node_ids[idx1]]][nodeB_id[node_ids[idx2]]] is not None:
                    if len(sepset_change) == 0:
                        sepset_change = set(cgB.sepset[nodeB_id[node_ids[idx1]]][nodeB_id[node_ids[idx2]]])
                    else:
                        sepset_change |= set(cgB.sepset[nodeB_id[node_ids[idx1]]][nodeB_id[node_ids[idx2]]])
            
            if len(sepset_change) != 0:
                cg.sepset[idx1][idx2] = list(sepset_change)
    cg.smallest = max(cgA.smallest, cgB.smallest)
    cg.partitions = partitions
    print("merge end:", time.time()-tt)
    return cg

def skeleton_conditional(
    curr_partition,
    max_partition,
    depth,
    origin_size,
    cg: CausalGraph,
    node_ids: List[int],
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    no_of_var = len(node_ids)
    idx_of_var = list(range(no_of_var))
    cg.partitions = 0
    cg.smallest = no_of_var
    if no_of_var == 0:
        return cg
    while cg.max_degree() - 1 > depth:
        #adjust max_partition < 1 to generate more child partitions
        if cg.max_degree() - 1 > depth and partition_available(cg, node_ids) and max_partition < 1:
            max_partition+=1
            break
        depth += 1
        edge_removal = []
        for (x, y) in permutations(idx_of_var, 2):
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) <= depth - 1:
                continue
            if y not in Neigh_x:
                continue
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
            for S in combinations(Neigh_x_noy, depth):
                t = time.time()
                p = cg.ci_test(node_ids[x], node_ids[y], list(np.array(node_ids)[list(S)]))
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
                            sepsets.add(node_ids[s])
                else:
                    if verbose:
                        print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
            append_value(cg.sepset, x, y, tuple(sepsets))
            append_value(cg.sepset, y, x, tuple(sepsets))

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)
    cg.partitions = 0
    if curr_partition < max_partition:
        cg = skeleton_partition(curr_partition, max_partition, depth, origin_size, cg, node_ids, alpha, indep_test, stable, background_knowledge, verbose, show_progress, node_names)
    return cg