#!/usr/bin/env bash
# Usage: logfiles_to_csv.sh [dirname]

# cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
LOGFILE="/dev/null"
if [[ -d $1 ]]; then LOGFILE="$1/results.csv"; fi;

COLS="time epochs val_loss val_grapheme_root_accuracy val_vowel_diacritic_accuracy val_consonant_diacritic_accuracy"
(
  echo "`echo $COLS | tr ' ' ','`,file"
  (
    for file in `find $* -type f -name '*.log'`; do
      (
        for key in $COLS;
          do grep "$key\":" $file | sed 's/^.*://' | perl -p -e 's/[,\s]*$/,/';
        done;
        echo $file | sed 's/^.*\///g'
        # echo "$file,`echo $file | sed 's/^.*\///g' | tr '-' ','`";
      ) | tr '\n' ' '; echo;
    done
  ) |
    tr '"' ' ' |
    tr ':' ',' |
    perl -p -e 's/ *, */,/g;' | perl -p -e 's/^ *|[ ,]*$//g;' |
    sort -t',' -k3
) |
  tee $LOGFILE

if [[ -d $1 ]]; then echo -e "----------\nWrote: $LOGFILE" > /dev/stderr; fi
