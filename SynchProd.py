import mkl
import ctypes
# print(mkl.__version__)
# import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.log.obj import Trace, Event
# from pm4py.objects.petri_net.sync_product import construct
# from pm4py.algo.conformance.alignments.petri_net import get_best_worst_cost, apply as align
from pm4py.algo.conformance.alignments.petri_net import algorithm as align

# Step 1: Define the Petri net
net = PetriNet('Net')

# Add places
p1 = PetriNet.Place('p1')
p2 = PetriNet.Place('p2')
p3 = PetriNet.Place('p3')
net.places.add(p1)
net.places.add(p2)
net.places.add(p3)

# Add transitions
t1 = PetriNet.Transition('t1', 'A')
t2 = PetriNet.Transition('t2', 'B')
net.transitions.add(t1)
net.transitions.add(t2)

# Add arcs
add_arc_from_to(p1, t1, net)
add_arc_from_to(t1, p2, net)
add_arc_from_to(p2, t2, net)
add_arc_from_to(t2, p3, net)

# Initial and final marking
initial_marking = Marking()
final_marking = Marking()
initial_marking[p1] = 1
final_marking[p3] = 1

# Step 2: Define the trace
trace = Trace()
trace.append(Event({'concept:name': 'A'}))
trace.append(Event({'concept:name': 'A'}))
trace.append(Event({'concept:name': 'B'}))

# # Step 3: Compute synchronous product
# sync_net, sync_initial_marking, sync_final_marking = construct(net, initial_marking, final_marking, trace)

# Parameters for alignment
parameters = {"ret_tuple_as_trans_desc": True}


# Step 3: Compute alignment
def main(trace, net):
    aligned_trace = align(trace, net, initial_marking=p1, final_marking = p2, parameters=parameters)
    return aligned_trace

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    # print(aligned_trace)