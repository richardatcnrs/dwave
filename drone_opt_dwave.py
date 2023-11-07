import dimod
import dwave_networkx as dnx
import networkx as nx
import dwave.embedding
from dwave.system import LeapHybridCQMSampler 

import sys
from itertools import groupby


def read_input():
    input_file = open(sys.argv[1], 'r')

    n = int(input_file.readline().strip())
    weight_matrix = [[None for i in range(n)] for y in range(n)]

    for i in range(n):
        weights = input_file.readline().split()
        for j in range(n):
            weight_matrix[i][j] = int(weights[j].strip())

    

    replaced_edges = eval(input_file.readline().strip())
    input_file.close()
    return n,weight_matrix,replaced_edges

def decode_solution(solutions, n, replaced_edges):
    #print(solution.energy, solution.values)
    post_decode = (q.decode(solution.values))

    # verify the correctness of solution
    for x in range(1, n):
        t_sum = 0
        for t in range(1,n):
            t_sum += post_decode[x][t]
        if not(t_sum == 1):
            print("solution is not permutation")
           
        

    for t in range(1, n):
        x_sum = 0
        for x in range(1,n):
            x_sum += post_decode[x][t]
        if not(x_sum == 1):
            print("solution is not permutation")
    
    # decode path
    path = [0]
    for t in range(1, n):
        for x in range(1, n):
            if post_decode[x][t] == 1:
                path.append(x)
                break
   

    
    print('Optimal cost =', solution.energy)
    
    # convert replaced edges
    
    if len(replaced_edges) > 0:
        converted_path = [0]
        for i in range(len(path)-1):
            if (path[i], path[i+1]) in replaced_edges:
                converted_path.extend(replaced_edges[path[i],path[i+1]])
                
               
            else:
                converted_path.append(path[i]) 
                converted_path.append(path[i+1])
        result = [i[0] for i in groupby(converted_path)]
        print("Optimal path =", result)
    else:
        print('Optimal path =', path)

def build_model(n, weight_matrix, replaced_edges):
    model = dimod.ConstrainedQuadraticModel()
    #var_list = dimod.variables.Variables(['x_' + str(i) + '_' + str(j) for i in range(1,n) for j in range(1,n)])
    model.add_variables('BINARY', ['x_' + str(i) + '_' + str(j) for i in range(1,n) for j in range(1,n)])
    
    # set up constraints
    for i in range(1, n):
        terms = []
        for j in range(1,n):
            terms.append(['x_' + str(i) + '_' + str(j),1])
       

        #print(terms)
        model.add_constraint_from_iterable(terms, '==', rhs = 1)
        #model.add_constraint_from_iterable([('x_2_3', 'x_4_5',1)], '==', rhs = 1)
                
    for j in range(1, n):
        terms = []
        for i in range(1,n):
            terms.append(['x_' + str(i) + '_' + str(j),1])
       

        #print(terms)
        model.add_constraint_from_iterable(terms, '==', rhs = 1)

    #print(len(model.constraints))
    
    obj = []
    
    # encode the 1st step
    for v in range(1,n):
        obj.append(['x_' + str(v) + '_1', weight_matrix[0][v]])

    # encode the hamiltonian path
    for t in range(1, n-1):
        for v in range(1,n):
            for v_p in range(1,n):
                if not(v == v_p):
                    obj.append(['x_' + str(v) + '_' + str(t), 'x_' + str(v_p) + '_' + str(t+1), weight_matrix[v][v_p]])

    model.set_objective(obj)

    #    f += q[v,1] * weight_matrix[0][v]
    #print(f)
    #for t in range(1, n-1):
    #    for v in range(1,n):
    #        for v_p in range(1,n):
    #            if not(v == v_p):
    #                f += weight_matrix[v][v_p]*q[v,t]*q[v_p,t+1]
    #print(f)
    #model = f


    return model

def run_cqm_and_collect_solutions(model, sampler):
    sampleset = sampler.sample_cqm(model, 30)
    return sampleset

def process_solutions(sampleset, n, weight_matrix, replaced_edges):
    perm_solutions = []
    for solution in sampleset:
        if check_perm(solution, n) == True:
            perm_solutions.append(solution)
    if len(perm_solutions) == 0:
        print('No valid solution')
        return 0,0
    elif len(perm_solutions) == 1:
        return compute_energy(perm_solutions[0], weight_matrix)
    else:
        min_solution = perm_solutions[0]
        min_path, min_energy = compute_energy(min_solution, weight_matrix)
    
        for i in range(1,len(perm_solutions)):
            current_path, current_energy = compute_energy(perm_solutions[i], weight_matrix)
            if current_energy < min_energy:
                min_energy = current_energy
                min_path = current_path
        
        # replace edges if needed
        if len(replaced_edges) > 0:
            converted_path = [0]
            for i in range(len(min_path)-1):
                if (min_path[i], min_path[i+1]) in replaced_edges:
                    converted_path.extend(replaced_edges[min_path[i],min_path[i+1]])
                
               
                else:
                    converted_path.append(min_path[i]) 
                    converted_path.append(min_path[i+1])
            min_path = [i[0] for i in groupby(converted_path)]
        return min_path, min_energy 

def compute_energy(solution, weight_matrix):
    cost = 0
    path = [0]
    for i in range(1,n):
        if solution['x_{}_1'.format(i)] == 1:
            path.append(i)
            cost += weight_matrix[0][i]
            break
    for t in range(2, n):
        for i in range(1,n):
            if solution['x_{}_{}'.format(i,t)] == 1:
                path.append(i)
                break
    for i in range(1, n-1):
        cost += weight_matrix[path[i]][path[i+1]]
    
    
    return path, cost

def check_perm(solution, n):
    for i in range(1, n):
        if not(sum(solution['x_{}_'.format(i) + str(j)] for j in range(1,n)) == 1):
            return False

    for j in range(1, n):
        if not(sum(solution['x_{}_'.format(i) + str(j)] for i in range(1,n)) == 1):
            return False

    return True

#client = FixstarsClient()
token_file = open('/home/richard/Desktop/data/dwave_token','r')
token = token_file.readline()
#print(token)
token_file.close()

n, weight_matrix, replaced_edges = read_input()


model = build_model(n, weight_matrix, replaced_edges)

sampler = LeapHybridCQMSampler()
#print(sampler.properties)
sampleset = run_cqm_and_collect_solutions(model, sampler)


min_path, min_energy = process_solutions(sampleset, n, weight_matrix, replaced_edges)
print('Optimal cost =', min_energy)
print('Optimal path =', min_path)





#print(sampler.min_time_limit(model))

#P16 = dnx.pegasus_graph(16)
#classical_sampler = SASampler()
#sampler = dimod.StructureComposite(classical_sampler, P16.nodes, P16.edges)
#h = {v: 0.0 for v in P16.nodes}
#J = {(u,v): 1 for u,v in P16.edges}
#sampleset = sampler.sample_ising(h,J)
#print(sampleset)

#print(n)
#print(weight_matrix)
#print(replaced_edges)


# setting up the variables
# need to encode permutation of n-1
#gen = SymbolGenerator(BinaryPoly)
#q = gen.array((n-1)*(n-1))
#q = gen.array(shape=(n,n))

#print(q)
#vars_map = {}
#index = 0

#for i in range(1,n):
#   for j in range(1,n):
#        vars_map[i,j] = q[index]
#        index += 1

#print(q)
#print(vars_map)

#f = BinaryPoly()




# constraint_1 sum_t x_v,t = 1 hard-code

#a = 2 * max(max(weight_matrix[i]) for i in range(n))
#b = 2 * max(max(weight_matrix[i]) for i in range(n))

#for v in range(1, n):
#    for t in range(1, n):
#        for t_p in range(1,n):
#            f += a*q[v,t]*q[v,t_p]
#        f -= 2*a*q[v,t]
#    f += a

#print(a,b)

# constraint_2 sum_v x_v,t = 1

#for t in range(1,n):
#    for v in range(1,n):
#        for v_p in range(1,n):
#            f += b*q[v,t]*q[v_p,t]
#        f -= 2*b*q[v,t]
#    f += b

# add constraint using FS API
#penalties = []
#for t in range(1,n):
#    temp = sum(q[v,t] for v in range(1,n))
#    p = equal_to(temp,1)
#    penalties.append(p)
    #print(p)
#for v in range(1,n):
#    temp = sum(q[v,t] for t in range(1,n))
#    p = equal_to(temp,1)
#    penalties.append(p)


# encoding the first step in the route

#for v in range(1,n):
#    f += q[v,1] * weight_matrix[0][v]

#print(f)
#for t in range(1, n-1):
#    for v in range(1,n):
#        for v_p in range(1,n):
#            if not(v == v_p):
#                f += weight_matrix[v][v_p]*q[v,t]*q[v_p,t+1]
#print(f)
#model = f

#for p in penalties:
#    temp = a*p
    #print(temp)
#    model = model + temp
#print(model)

#client = DWaveSamplerClient()
#client.token = token.strip()
#client.solver = 'Advantage_system6.3'
#client.parameters.num_reads = 100
#f = 2*q[2][2]*q[2][5] - q[3][5] - q[4][0] + 1

#result = client.solve(f)

#print(client.solver_names)

#client.parameters.timeout = 30000
#print(client.parameters)
#print('client token',client.token)
#solver = Solver(client)
#solver.filter_solution = False
#print(solver)
#print(solver.chain_strength)
#print(solver.client)
#print(solver.client_result)
#print(model)


#print(model.input_constraints)
#print(model.input_poly)

#result = solver.solve(model)
#print('execution time =',solver.execution_time)
#print(len(result))
#for solution in result:
#    print(solution)
    #decode_solution(solution, n, replaced_edges)

