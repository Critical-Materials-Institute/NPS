KEY='Loss'

for i in $@; do
  if [ -f $i ]; then
    f=$i
  else
    f=$i
  fi
  if  true ; then
    echo -n "'<grep --text \"$KEY\" \"$f\"|awk \"{print \\\$8}\" ' u 1 w l t \"$f\",";
#    echo -n "'<grep = \"$f\"' u 1:3 w l t \"$f\","
  else
    echo -n "'<awk \"NR>12{print \\\$2,\\\$3}\" \"$f\"' u 1:2 w p t \"$f\",";
  fi
done |sed 's/^/set ylabel "loss"\nset xlabel "iter"\nset logscale y; set yrange [0.006:0.02]; plot /;s/,$/\npause 99\n/' | gnuplot 

