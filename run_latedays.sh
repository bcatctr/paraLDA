#!/bin/bash

# Max allowed nodes.
MAX_ALLOWED_NODES=17

# Each node has 24 cores.
TOTAL_PROCESSORS_PER_NODE=24

# 2 minute time limit.
WALLTIME=1

# Ensure 2 arguments for nodes and processors per node.
if [ $# -ne 2 ]; then
  echo "Usage: $(basename $0) nodes processors_per_node"
  exit $E_BADARGS
fi

# Get command line arguments.
NODES=$1
PROCESSORS_PER_NODE=$2

# Validate arguments.
if [ $NODES -le 0 ] || [ $NODES -gt $MAX_ALLOWED_NODES ]; then
    echo "ERROR: Only $MAX_ALLOWED_NODES nodes allowed."
    exit $E_BADARGS
fi
if [ $PROCESSORS_PER_NODE -le 0 ] || [ $PROCESSORS_PER_NODE -gt $TOTAL_PROCESSORS_PER_NODE ]; then
    echo "ERROR: Each node only has $TOTAL_PROCESSORS_PER_NODE cores."
    exit $E_BADARGS
fi
if [ ! -f "./bin/paraLDA" ]; then
    echo "ERROR: ./bin/paraLDA program does not exist."
    exit $E_BADARGS
fi

# Submit the job.  No need to modify this.
qsub -l walltime=0:$WALLTIME:00,nodes=$NODES:ppn=$TOTAL_PROCESSORS_PER_NODE -F "$NODES $PROCESSORS_PER_NODE" latedays.qsub
