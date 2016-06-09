#!/usr/bin/perl 

use strict; 
use warnings;

for (1..99) { print $_." Luftballons\n"; }

# This is a commented line

my $string = "Hello World!";

print $string."\n\n";

$string =~ s/Hello/Goodbye/;

print $string."\n\n";

test();

exit;

sub test { print "All good.\n"; }