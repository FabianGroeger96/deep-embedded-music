import sys
import trace
from src.train_triplet_loss import main

if __name__ == "__main__":
    # create a Trace object, telling it what to ignore, and whether to
    # do tracing or line-counting or both.
    tracer = trace.Trace(
        ignoredirs=[sys.prefix, sys.exec_prefix],
        trace=True,
        count=False)

    # run the new command using the given tracer
    tracer.run('main()')

    # make a report, placing output in the current directory
    # r = tracer.results()
    # r.write_results(show_missing=True, coverdir=".")
