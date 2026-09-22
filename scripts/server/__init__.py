"""Server-side engineering harness for DATA-28.

Hardware-side measurements of the M1 baseline zoo and a concurrency sweep over
N parallel workers. The scripts in this package only touch validation data and
never the test split; they are read against the M1 baseline implementations
that live in ``drososense.baselines``.
"""