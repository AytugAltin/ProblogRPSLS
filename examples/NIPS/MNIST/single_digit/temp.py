from pysdd.sdd import SddManager, Vtree, WmcManager
vtree = Vtree(var_count=4, var_order=[2,1,4,3], vtree_type="balanced")
sdd = SddManager.from_vtree(vtree)
a, b, c, d = sdd.vars

# Build SDD for formula
formula = (a & b) | (b & c) | (c & d)

# Model Counting
wmc = formula.wmc(log_mode=False)
print(f"Model Count: {wmc.propagate()}")
wmc.set_literal_weight(a, 0.5)
print(f"Weighted Model Count: {wmc.propagate()}")

# Visualize SDD and Vtree
with open("output/sdd.dot", "w") as out:
    print(formula.dot(), file=out)
with open("output/vtree.dot", "w") as out:
    print(vtree.dot(), file=out)
    