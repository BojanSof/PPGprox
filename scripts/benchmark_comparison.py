# compare the filtering and FFT outputs between CMSIS DSP and SciPy,
# calculate absolute and relative errors
import warnings

import numpy as np
from scipy import signal as sig
from scipy import fft as fft

from plot_hexdump import convert_hexdump
from sine_wave_gen import generate_sine_wave

warnings.simplefilter("ignore")


def calculate_max_abs_error(arr1, arr2):
    return np.amax(np.abs(arr1 - arr2))


def calculate_max_rel_error(arr1, arr2):
    return np.amax(np.nan_to_num(np.abs((arr1 - arr2) / arr1)))


# input signal
hexdump_input = r"""
00 00 00 00 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |....MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 21 80 c0 ab  d7 09 82 43 83 08 05 43 |....!... ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  66 e9 52 ac 4d 47 9b 43 |N.[.MG.. f.R.MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 32 40 57 ac |..gC.... ....2@W.
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
66 e9 d2 ac 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |f...MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 b2 fe 1e 2b  d7 09 82 43 83 08 05 43 |.......+ ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  1a de ef ac 4d 47 9b 43 |N.[.MG.. ....MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 3a a0 8e ac |..gC.... ....:...
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
66 e9 52 ad 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |f.R.MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 2d 81 8b ab  d7 09 82 43 83 08 05 43 |....-... ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  f5 3f 5c 29 4d 47 9b 43 |N.[.MG.. .?\)MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 2e 90 7c ad |..gC.... ......|.
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
1a de 6f ad 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |..o.MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 93 1f dc 2c  d7 09 82 43 83 08 05 43 |......., ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  73 58 3c ad 4d 47 9b 43 |N.[.MG.. sX<.MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 7d a0 eb ac |..gC.... ....}...
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
66 e9 d2 ad 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |f...MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 1c c1 66 ac  d7 09 82 43 83 08 05 43 |......f. ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  9a 34 3f ac 4d 47 9b 43 |N.[.MG.. .4?.MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 f5 04 21 ab |..gC.... ......!.
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
f5 3f dc 29 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |.?.)MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 16 7c 04 ae  d7 09 82 43 83 08 05 43 |.....|.. ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  76 50 04 ae 4d 47 9b 43 |N.[.MG.. vP..MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 40 5f f1 2c |..gC.... ....@_.,
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
1a de ef ad 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |....MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 34 d8 ca ad  d7 09 82 43 83 08 05 43 |....4... ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  73 c9 41 2d 4d 47 9b 43 |N.[.MG.. s.A-MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 38 08 b4 ad |..gC.... ....8...
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
73 58 bc ad 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |sX..MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 c3 c7 9f 2d  d7 09 82 43 83 08 05 43 |.......- ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  a0 15 97 ad 4d 47 9b 43 |N.[.MG.. ....MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 81 d0 52 ad |..gC.... ......R.
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
66 e9 52 ae 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |f.R.MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 bb 67 cf 2d  d7 09 82 43 83 08 05 43 |.....g.- ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  f3 1f 2e ad 4d 47 9b 43 |N.[.MG.. ....MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 25 a4 27 ae |..gC.... ....%.'.
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
9a 34 bf ac 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |.4..MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
d7 09 82 c3 27 3c 1c ae  d7 09 82 43 83 08 05 43 |....'<.. ...C...C
d5 94 67 c3 7c 19 ad c3  36 1d 6b c2 aa 91 65 43 |..g.|... 6.k...eC
69 6f eb 42 c4 55 72 c3  e4 8b b5 c3 18 36 be c2 |io.B.Ur. .....6..
a8 61 2e 43 f2 65 68 42  2a 49 93 c3 50 20 c6 c3 |.a.C.ehB *I..P ..
18 36 be c2 80 8a 4f 43  19 ac dc 42 1c f4 6a c3 |.6....OC ...B..j.
4f 88 aa c3 36 1d 6b c2  51 6f 60 43 f7 2d f2 42 |O...6.k. Qo`C.-.B
4e a3 5b c3 4d 47 9b c3  34 a5 a8 ab 4d 47 9b 43 |N.[.MG.. 4...MG.C
4e a3 5b 43 f7 2d f2 c2  51 6f 60 c3 36 1d 6b 42 |N.[C.-.. Qo`.6.kB
4f 88 aa 43 1c f4 6a 43  19 ac dc c2 80 8a 4f c3 |O..C..jC ......O.
18 36 be 42 50 20 c6 43  2a 49 93 43 f2 65 68 c2 |.6.BP .C *I.C.eh.
a8 61 2e c3 18 36 be 42  e4 8b b5 43 c4 55 72 43 |.a...6.B ...C.UrC
69 6f eb c2 aa 91 65 c3  36 1d 6b 42 7c 19 ad 43 |io....e. 6.kB|..C
d5 94 67 43 83 08 05 c3  d7 09 82 c3 d7 2b 12 2e |..gC.... .....+..
d7 09 82 43 83 08 05 43  d5 94 67 c3 7c 19 ad c3 |...C...C ..g.|...
36 1d 6b c2 aa 91 65 43  69 6f eb 42 c4 55 72 c3 |6.k...eC io.B.Ur.
e4 8b b5 c3 18 36 be c2  a8 61 2e 43 f2 65 68 42 |.....6.. .a.C.ehB
2a 49 93 c3 50 20 c6 c3  18 36 be c2 80 8a 4f 43 |*I..P .. .6....OC
19 ac dc 42 1c f4 6a c3  4f 88 aa c3 36 1d 6b c2 |...B..j. O...6.k.
51 6f 60 43 f7 2d f2 42  4e a3 5b c3 4d 47 9b c3 |Qo`C.-.B N.[.MG..
f5 3f 5c 2a 4d 47 9b 43  4e a3 5b 43 f7 2d f2 c2 |.?\*MG.C N.[C.-..
51 6f 60 c3 36 1d 6b 42  4f 88 aa 43 1c f4 6a 43 |Qo`.6.kB O..C..jC
19 ac dc c2 80 8a 4f c3  18 36 be 42 50 20 c6 43 |......O. .6.BP .C
2a 49 93 43 f2 65 68 c2  a8 61 2e c3 18 36 be 42 |*I.C.eh. .a...6.B
e4 8b b5 43 c4 55 72 43  69 6f eb c2 aa 91 65 c3 |...C.UrC io....e.
36 1d 6b 42 7c 19 ad 43  d5 94 67 43 83 08 05 c3 |6.kB|..C ..gC....
"""
input_sig = convert_hexdump(hexdump_input)

# filtering 1 (sample by sample)
hexdump1 = r"""
00 00 00 00 aa 23 f1 42  02 f8 3b 43 43 b4 09 c2 |.....#.B ..;CC...
8c be 73 c3 38 16 02 c3  3a 3a 23 43 db 81 6c 43 |..s.8... ::#C..lC
a8 97 24 c1 a2 6b 69 c3  52 85 e9 c2 47 23 3b 43 |..$..ki. R...G#;C
ec 77 83 43 fc 81 b7 40  21 f4 6b c3 8d c7 0e c3 |.w.C...@ !.k.....
56 cb 0b 43 a4 25 50 43  2b 7e 22 c2 50 2b 84 c3 |V..C.%PC +~".P+..
58 8d 13 c3 e0 f0 1c 43  52 bf 69 43 54 4d b0 c1 |X......C R.iCTM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
"""
output1 = convert_hexdump(hexdump1)

# filtering 2 (whole array)
hexdump2 = r"""
00 00 00 00 aa 23 f1 42  02 f8 3b 43 43 b4 09 c2 |.....#.B ..;CC...
8c be 73 c3 38 16 02 c3  3a 3a 23 43 db 81 6c 43 |..s.8... ::#C..lC
a8 97 24 c1 a2 6b 69 c3  52 85 e9 c2 47 23 3b 43 |..$..ki. R...G#;C
ec 77 83 43 fc 81 b7 40  21 f4 6b c3 8d c7 0e c3 |.w.C...@ !.k.....
56 cb 0b 43 a4 25 50 43  2b 7e 22 c2 50 2b 84 c3 |V..C.%PC +~".P+..
58 8d 13 c3 e0 f0 1c 43  52 bf 69 43 54 4d b0 c1 |X......C R.iCTM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
25 f3 82 c3 c7 80 26 c3  30 53 ed 42 64 f0 3d 43 |%.....&. 0S.Bd.=C
12 7b 5e c2 98 f1 89 c3  8c 78 1b c3 0a c7 18 43 |.{^..... .x.....C
ae 67 69 43 c4 6d 94 c1  d3 a4 7e c3 17 97 1b c3 |.giC.m.. ..~.....
71 0f 05 43 47 98 4f 43  80 f0 0b c2 1a 99 7c c3 |q..CG.OC ......|.
0a e5 01 c3 b6 3c 34 43  54 28 83 43 70 9d 35 41 |.....<4C T(.Cp.5A
6c 37 60 c3 5f 3d fa c2  d6 17 23 43 6a b7 6c 43 |l7`._=.. ..#Cj.lC
30 7a e7 c0 ab a7 62 c3  71 56 d4 c2 1e 5f 49 43 |0z....b. qV..._IC
2b 43 8c 43 3a ca d2 41  c5 aa 54 c3 e7 66 ea c2 |+C.C:..A ..T..f..
ae 41 27 43 0d 0f 6d 43  48 7c 2b c1 22 e9 69 c3 |.A'C..mC H|+.".i.
d2 29 ea c2 45 f9 3a 43  3a 6f 83 43 5c d4 b6 40 |.)..E.:C :o.C\..@
dd f4 6b c3 f7 c6 0e c3  00 cc 0b 43 13 26 50 43 |..k..... ...C.&PC
4b 7d 22 c2 45 2b 84 c3  50 8d 13 c3 e2 f0 1c 43 |K}".E+.. P......C
53 bf 69 43 4c 4d b0 c1  25 f3 82 c3 c7 80 26 c3 |S.iCLM.. %.....&.
30 53 ed 42 64 f0 3d 43  12 7b 5e c2 98 f1 89 c3 |0S.Bd.=C .{^.....
8c 78 1b c3 0a c7 18 43  ae 67 69 43 c4 6d 94 c1 |.x.....C .giC.m..
d3 a4 7e c3 17 97 1b c3  71 0f 05 43 47 98 4f 43 |..~..... q..CG.OC
80 f0 0b c2 1a 99 7c c3  0a e5 01 c3 b6 3c 34 43 |......|. .....<4C
54 28 83 43 70 9d 35 41  6c 37 60 c3 5f 3d fa c2 |T(.Cp.5A l7`._=..
d6 17 23 43 6a b7 6c 43  30 7a e7 c0 ab a7 62 c3 |..#Cj.lC 0z....b.
71 56 d4 c2 1e 5f 49 43  2b 43 8c 43 3a ca d2 41 |qV..._IC +C.C:..A
c5 aa 54 c3 e7 66 ea c2  ae 41 27 43 0d 0f 6d 43 |..T..f.. .A'C..mC
48 7c 2b c1 22 e9 69 c3  d2 29 ea c2 45 f9 3a 43 |H|+.".i. .)..E.:C
3a 6f 83 43 5c d4 b6 40  dd f4 6b c3 f7 c6 0e c3 |:o.C\..@ ..k.....
00 cc 0b 43 13 26 50 43  4b 7d 22 c2 45 2b 84 c3 |...C.&PC K}".E+..
50 8d 13 c3 e2 f0 1c 43  53 bf 69 43 4c 4d b0 c1 |P......C S.iCLM..
"""
output2 = convert_hexdump(hexdump2)

# check that output1 == output2
print(f"Filtering 1 == Filtering 2 : {all(output1 == output2)}")
output_cmsis = output1

# perform filtering with scipy
sos = sig.butter(1, [3, 12], btype='bandpass', fs=50, output='sos')
output_python = sig.sosfilt(sos, input_sig)

# calculate max absolute and max relative error
filter_abs_err = calculate_max_abs_error(output_python, output_cmsis)
print(f"Filtering maximum absolute error: {filter_abs_err}")
filter_rel_err = calculate_max_rel_error(output_python, output_cmsis)
print(f"Filtering maximum relative error: {filter_rel_err}")

# FFT test
hexdump_fft = r"""
f9 f9 ee 44 e0 74 28 42  80 73 ef 44 cf 02 74 c0 |...D.t(B .s.D..t.
9c e3 f0 44 80 3d f6 c0  08 55 f3 44 0e 8c 3b c1 |...D.=.. .U.D..;.
b0 da f6 44 35 97 7f c1  cc 90 fb 44 ff 5c a4 c1 |...D5... ...D.\..
d7 cf 00 45 df 5c cc c1  86 9f 04 45 1d e3 f8 c1 |...E.\.. ...E....
d4 5d 09 45 e4 a9 15 c2  ea 3f 0f 45 4b c2 32 c2 |.].E.... .?.EK.2.
96 90 16 45 42 f9 54 c2  65 bb 1f 45 b9 0f 7e c2 |...EB.T. e..E..~.
bf 5f 2b 45 b1 4a 98 c2  78 73 3a 45 59 38 b8 c2 |._+E.J.. xs:EY8..
2f 85 4e 45 da fc e1 c2  94 47 6a 45 c2 83 0d c3 |/.NE.... .GjE....
9e 6b 89 45 1c d4 36 c3  0c 99 a9 45 31 02 78 c3 |.k.E..6. ...E1.x.
36 09 e4 45 12 fd b6 c3  42 f6 36 46 df f4 20 c4 |6..E.... B.6F.. .
99 f6 06 47 99 10 02 c5  21 3f ee c6 b2 66 fb 44 |...G.... !?...f.D
86 b9 1b c6 8e e4 33 44  cb 45 b3 c5 e4 bb e2 43 |......3D .E.....C
9c a9 74 c5 60 71 a9 43  64 5d 35 c5 03 a3 89 43 |..t.`q.C d]5....C
fc 25 0d c5 f2 f0 6a 43  78 c4 e2 c4 c0 2f 4f 43 |.%....jC x..../OC
20 31 ba c4 fa ff 3a 43  da 4a 9b c4 96 b8 2b 43 | 1....:C .J....+C
76 02 83 c4 2d cf 1f 43  5e e5 5e c4 c3 4e 16 43 |v...-..C ^.^..N.C
cd bd 3e c4 be 97 0e 43  7a df 23 c4 22 3d 08 43 |..>....C z.#."=.C
44 1b 0d c4 46 f2 02 43  67 2b f3 c3 d7 fe fc 42 |D...F..C g+.....B
f2 57 d1 c3 d6 77 f5 42  90 c5 b3 c3 8f 0e ef 42 |.W...w.B .......B
76 b6 99 c3 18 95 e9 42  67 95 82 c3 66 e3 e4 42 |v......B g...f..B
98 d8 5b c3 46 dc e0 42  f0 b7 36 c3 8b 66 dd 42 |..[.F..B ..6..f.B
36 2e 15 c3 74 6e da 42  1a 78 ed c2 0a e4 d7 42 |6...tn.B .x.....B
de f0 b5 c2 86 b8 d5 42  20 16 83 c2 fc e1 d3 42 |.......B  ......B
0a a7 28 c2 f0 54 d2 42  4e ac a4 c1 0f 0a d1 42 |..(..T.B N......B
c0 4b 18 bf ba f9 cf 42  5f e0 8f 41 91 1d cf 42 |.K.....B _..A...B
a6 3e 0d 42 8d 70 ce 42  ed 16 4e 42 59 ef cd 42 |.>.B.p.B ..NBY..B
50 76 85 42 ca 90 cd 42  90 19 a2 42 0f 56 cd 42 |Pv.B...B ...B.V.B
19 22 bd 42 ee 39 cd 42  7d ba d6 42 37 39 cd 42 |.".B.9.B }..B79.B
80 08 ef 42 23 51 cd 42  c6 16 03 43 ff 7e cd 42 |...B#Q.B ...C.~.B
12 24 0e 43 92 c0 cd 42  db b9 18 43 3c 13 ce 42 |.$.C...B ...C<..B
e0 e4 22 43 c7 74 ce 42  8a b0 2c 43 40 e4 ce 42 |.."C.t.B ..,C@..B
2a 27 36 43 23 5d cf 42  68 53 3f 43 6d df cf 42 |*'6C#].B hS?Cm..B
31 3e 48 43 4e 68 d0 42  6e f0 50 43 fc f5 d0 42 |1>HCNh.B n.PC...B
82 72 59 43 f5 85 d1 42  a2 cc 61 43 3a 16 d2 42 |.rYC...B ..aC:..B
72 06 6a 43 8a a4 d2 42  10 28 72 43 1e 2e d3 42 |r.jC...B .(rC...B
53 39 7a 43 0e b1 d3 42  e5 20 81 43 a4 29 d4 42 |S9zC...B . .C.).B
e4 24 85 43 06 95 d4 42  c9 2c 89 43 a4 ef d4 42 |.$.C...B .,.C...B
17 3d 8d 43 90 35 d5 42  5a 5a 91 43 fc 62 d5 42 |.=.C.5.B ZZ.C.b.B
d7 89 95 43 53 71 d5 42  b8 d0 99 43 65 5e d5 42 |...CSq.B ...Ce^.B
5f 36 9e 43 15 1d d5 42  88 c0 a2 43 28 ab d4 42 |_6.C...B ...C(..B
b6 77 a7 43 1c fe d3 42  1a 65 ac 43 d2 0a d3 42 |.w.C...B .e.C...B
5f 93 b1 43 bc c4 d1 42  9b 0f b7 43 e5 1c d0 42 |_..C...B ...C...B
a7 e8 bc 43 bb fe cd 42  91 31 c3 43 da 55 cb 42 |...C...B .1.C.U.B
db 00 ca 43 da 02 c8 42  92 73 d1 43 14 e3 c3 42 |...C...B .s.C...B
f2 ad d9 43 82 c7 be 42  06 df e2 43 75 74 b8 42 |...C...B ...Cut.B
dc 44 ed 43 e9 99 b0 42  fe 32 f9 43 51 cc a6 42 |.D.C...B .2.CQ..B
6a 8e 03 44 b2 76 9a 42  49 d3 0b 44 93 c4 8a 42 |j..D.v.B I..D...B
78 e1 15 44 ac f6 6c 42  55 73 22 44 14 6f 37 42 |x..D..lB Us"D.o7B
e0 b7 32 44 70 99 dd 41  4a be 48 44 80 d4 05 40 |..2Dp..A J.HD...@
e8 74 68 44 98 4d 10 c2  c7 34 8d 44 14 d5 c3 c2 |.thD.M.. .4.D....
e4 b5 ba 44 fe 12 55 c3  9e 5d 14 45 9c f2 f7 c3 |...D..U. .].E....
24 ca ee 45 d8 c2 09 c5  ef 02 91 c5 ab b3 df 44 |$..E.... .......D
1c d3 c3 c4 7e 8b 46 44  f4 cc 55 c4 ea 97 0d 44 |....~.FD ..U....D
6a 62 06 c4 34 46 e9 43  24 69 b2 c3 bc a2 cd 43 |jb..4F.C $i.....C
ba af 6f c3 80 5a bc 43  da 39 1d c3 64 bb b0 43 |..o..Z.C .9..d..C
d2 7f bf c2 93 8c a8 43  cb b9 3f c2 96 9d a2 43 |.......C ..?....C
9c f7 17 c1 e2 3c 9e 43  2a b5 b1 41 34 fc 9a 43 |.....<.C *..A4..C
08 ca 43 42 3a 93 98 43  f8 ca 8f 42 cc d0 96 43 |..CB:..C ...B...C
26 cd b7 42 2b 92 95 43  1e 26 db 42 7b be 94 43 |&..B+..C .&.B{..C
7f bd fa 42 ca 42 94 43  5f a3 0b 43 62 11 94 43 |...B.B.C _..Cb..C
42 a5 18 43 76 1f 94 43  69 9a 24 43 9f 64 94 43 |B..Cv..C i.$C.d.C
25 ae 2f 43 53 da 94 43  64 03 3a 43 55 7b 95 43 |%./CS..C d.:CU{.C
ae b6 43 43 a3 43 96 43  ed df 4c 43 9a 2f 97 43 |..CC.C.C ..LC./.C
27 93 55 43 f8 3c 98 43  c0 e0 5d 43 4d 69 99 43 |'.UC.<.C ..]CMi.C
2a d7 65 43 df b2 9a 43  c6 82 6d 43 6b 18 9c 43 |*.eC...C ..mCk..C
47 ee 74 43 d3 98 9d 43  0c 23 7c 43 39 33 9f 43 |G.tC...C .#|C93.C
83 94 81 43 3f e7 a0 43  ff 03 85 43 25 b4 a2 43 |...C?..C ...C%..C
37 63 88 43 e2 99 a4 43  1a b5 8b 43 4a 98 a6 43 |7c.C...C ...CJ..C
65 fc 8e 43 64 af a8 43  90 3b 92 43 62 df aa 43 |e..Cd..C .;.Cb..C
00 75 95 43 87 28 ad 43  d4 aa 98 43 2a 8b af 43 |.u.C.(.C ...C*..C
06 df 9b 43 e3 07 b2 43  8e 13 9f 43 4d 9f b4 43 |...C...C ...CM..C
83 4a a2 43 83 51 b7 43  83 85 a5 43 41 20 ba 43 |.J.C.Q.C ...CA .C
e6 c6 a8 43 a3 0b bd 43  4d 0f ac 43 33 15 c0 43 |...C...C M..C3..C
2f 61 af 43 fa 3d c3 43  26 be b2 43 fc 86 c6 43 |/a.C.=.C &..C...C
28 28 b6 43 73 f1 c9 43  88 a0 b9 43 57 7f cd 43 |((.Cs..C ...CW..C
7c 29 bd 43 ec 31 d1 43  d5 c4 c0 43 0d 0b d5 43 ||).C.1.C ...C...C
69 74 c4 43 8e 0c d9 43  50 3a c8 43 99 38 dd 43 |it.C...C P:.C.8.C
a1 18 cc 43 74 91 e1 43  96 11 d0 43 86 19 e6 43 |...Ct..C ...C...C
a2 27 d4 43 8e d3 ea 43  3b 5d d8 43 78 c2 ef 43 |.'.C...C ;].Cx..C
f6 b4 dc 43 75 e9 f4 43  c8 31 e1 43 0e 4c fa 43 |...Cu..C .1.C.L.C
c5 d6 e5 43 02 ee ff 43  40 a7 ea 43 b7 e9 02 44 |...C...C @..C...D
b0 a6 ef 43 92 00 06 44  16 d9 f4 43 fd 3d 09 44 |...C...D ...C.=.D
b9 42 fa 43 ab a4 0c 44  45 e8 ff 43 cf 37 10 44 |.B.C...D E..C.7.D
58 e7 02 44 ac fa 13 44  d3 fd 05 44 07 f1 17 44 |X..D...D ...D...D
b0 3a 09 44 f3 1e 1c 44  56 a1 0c 44 20 89 20 44 |.:.D...D V..D . D
9c 35 10 44 9f 34 25 44  ba fb 13 44 37 27 2a 44 |.5.D.4%D ...D7'*D
76 f8 17 44 51 67 2f 44  24 31 1c 44 3a fc 34 44 |v..DQg/D $1.D:.4D
d3 ab 20 44 19 ee 3a 44  4f 6f 25 44 4c 46 41 44 |.. D..:D Oo%DLFAD
63 83 2a 44 67 0f 48 44  e7 f0 2f 44 90 55 4f 44 |c.*Dg.HD ../D.UOD
08 c2 35 44 b2 26 57 44  a2 02 3c 44 ff 92 5f 44 |..5D.&WD ..<D.._D
2b c0 42 44 11 ad 68 44  b6 0a 4a 44 d3 8a 72 44 |+.BD..hD ..JD..rD
ee f4 51 44 0a 46 7d 44  ff 94 5a 44 89 7e 84 44 |..QD.F}D ..ZD.~.D
46 05 64 44 16 ea 8a 44  96 65 6e 44 77 fb 91 44 |F.dD...D .enDw..D
34 dc 79 44 2b cd 99 44  2c 4c 83 44 88 7f a2 44 |4.yD+..D ,L.D...D
29 6a 8a 44 af 3a ac 44  81 6c 92 44 d7 30 b7 44 |)j.D.:.D .l.D.0.D
c3 81 9b 44 e4 a1 c3 44  a2 e5 a5 44 1f e0 d1 44 |...D...D ...D...D
ae e6 b1 44 9e 57 e2 44  12 ee bf 44 14 99 f5 44 |...D.W.D ...D...D
d9 8b d0 44 68 35 06 45  eb 8a e4 44 04 f2 13 45 |...Dh5.E ...D...E
6f 12 fd 44 78 cd 24 45  65 f0 0d 45 f8 fa 39 45 |o..Dx.$E e..E..9E
84 dd 21 45 7f 62 55 45  98 a7 3c 45 80 3c 7a 45 |..!E.bUE ..<E.<zE
d6 98 62 45 41 38 97 45  b5 3f 8e 45 8f 0e bf 45 |..bEA8.E .?.E...E
e2 de bf 45 51 ac 01 46  ee 4e 14 46 04 c1 49 46 |...EQ..F .N.F..IF
0b cd a5 46 10 04 e3 46  d0 bf a4 c7 48 0e e3 c7 |...F...F ....H...
87 41 5a c6 30 65 97 c6  46 8d ec c5 f6 2e 25 c6 |.AZ.0e.. F.....%.
c9 8f a1 c5 a3 28 e3 c5  fb 8d 74 c5 56 18 ad c5 |.....(.. ..t.V...
df 32 44 c5 c4 d2 8b c5  d0 6f 23 c5 d2 90 6a c5 |.2D..... .o#...j.
7c c5 0b c5 2a 01 4a c5  e4 bf f3 c4 ff 61 31 c5 ||...*.J. .....a1.
4e bb d7 c4 80 1c 1e c5  de 33 c1 c4 de 9d 0e c5 |N....... .3......
35 b1 ae c4 1c e3 01 c5  63 36 9f c4 43 7c ee c4 |5....... c6..C|..
1e 13 92 c4 45 6b dc c4  c6 c8 86 c4 f0 e4 cc c4 |....Ek.. ........
09 f4 79 c4 cf 68 bf c4  2e c1 68 c4 4a 96 b3 c4 |..y..h.. ..h.J...
36 8d 59 c4 f1 22 a9 c4  d4 03 4c c4 08 d5 9f c4 |6.Y..".. ..L.....
76 e2 3f c4 ac 7e 97 c4  b4 f3 34 c4 14 fb 8f c4 |v.?..~.. ..4.....
2e 0c 2b c4 6e 2c 89 c4  5b 08 22 c4 41 fa 82 c4 |..+.n,.. [.".A...
c8 ca 19 c4 b8 a0 7a c4  f6 3a 12 c4 a0 3b 70 c4 |......z. .:...;p.
24 44 0b c4 d8 a8 66 c4  ec d4 04 c4 4a d0 5d c4 |$D....f. ....J.].
a3 bc fd c3 64 9d 55 c4  17 a7 f2 c3 86 fe 4d c4 |....d.U. ......M.
01 53 e8 c3 89 e4 46 c4  50 ad de c3 2a 42 40 c4 |.S....F. P...*B@.
1b a5 d5 c3 02 0c 3a c4  b7 2b cd c3 c0 37 34 c4 |......:. .+...74.
a7 34 c5 c3 ba bc 2e c4  13 b4 bd c3 1b 93 29 c4 |.4...... ......).
0d a0 b6 c3 ef b3 24 c4  aa ef af c3 0b 19 20 c4 |......$. ...... .
d3 9a a9 c3 e4 bc 1b c4  6c 9a a3 c3 9e 9a 17 c4 |........ l.......
e0 e7 9d c3 b5 ad 13 c4  d5 7d 98 c3 4d f2 0f c4 |........ .}..M...
32 56 93 c3 a9 64 0c c4  f8 6c 8e c3 9d 01 09 c4 |2V...d.. .l......
8f bd 89 c3 2f c6 05 c4  10 44 85 c3 af af 02 c4 |..../... .D......
ff fc 80 c3 64 77 ff c3  e9 c9 79 c3 db cf f9 c3 |....dw.. ..y.....
37 f2 71 c3 c7 64 f4 c3  56 6d 6a c3 4a 32 ef c3 |7.q..d.. Vmj.J2..
13 36 63 c3 2f 35 ea c3  c4 47 5c c3 03 6a e5 c3 |.6c./5.. .G\..j..
68 9e 55 c3 04 ce e0 c3  d6 35 4f c3 69 5e dc c3 |h.U..... .5O.i^..
73 0a 49 c3 cc 18 d8 c3  d6 18 43 c3 c2 fa d3 c3 |s.I..... ..C.....
f1 5d 3d c3 3e 02 d0 c3  c3 d6 37 c3 53 2d cc c3 |.]=.>... ..7.S-..
56 80 32 c3 e6 79 c8 c3  88 58 2d c3 8a e6 c4 c3 |V.2..y.. .X-.....
c0 5c 28 c3 8b 71 c1 c3  db 8a 23 c3 5f 19 be c3 |.\(..q.. ..#._...
a6 e0 1e c3 d0 dc ba c3  6e 5c 1a c3 53 ba b7 c3 |........ n\..S...
e6 fb 15 c3 d0 b0 b4 c3  86 bd 11 c3 96 bf b1 c3 |........ ........
a8 a0 0d c3 65 e4 ae c3  63 a2 09 c3 46 1f ac c3 |....e... c...F...
b4 c1 05 c3 1e 6f a9 c3  5e fd 01 c3 f6 d2 a6 c3 |.....o.. ^.......
fb a7 fc c2 ff 49 a4 c3  c4 88 f5 c2 80 d3 a1 c3 |.....I.. ........
d4 9a ee c2 c5 6e 9f c3  03 dc e7 c2 04 1b 9d c3 |.....n.. ........
2d 49 e1 c2 ce d7 9a c3  5c e1 da c2 8a a4 98 c3 |-I...... \.......
f9 a1 d4 c2 a1 80 96 c3  50 89 ce c2 ab 6b 94 c3 |........ P....k..
32 95 c8 c2 35 65 92 c3  de c3 c2 c2 e5 6c 90 c3 |2...5e.. .....l..
9a 13 bd c2 4d 82 8e c3  cd 82 b7 c2 7b a5 8c c3 |....M... ....{...
d9 0d b2 c2 8a d5 8a c3  6d b3 ac c2 ce 12 89 c3 |........ m.......
50 73 a7 c2 73 5d 87 c3  e6 48 a2 c2 0a b5 85 c3 |Ps..s].. .H......
24 33 9d c2 d1 19 84 c3  98 2e 98 c2 1b 8c 82 c3 |$3...... ........
d0 37 93 c2 63 0c 81 c3  7c 4b 8e c2 24 36 7f c3 |.7..c... |K..$6..
82 64 89 c2 18 73 7c c3  aa 7d 84 c2 e0 d1 79 c3 |.d...s|. .}....y.
c0 1d 7f c2 d0 56 77 c3  30 1c 75 c2 c8 07 75 c3 |.....Vw. 0.u...u.
a0 da 6a c2 9c ed 72 c3  f0 2e 60 c2 a0 15 71 c3 |..j...r. ..`...q.
a0 da 54 c2 b8 94 6f c3  80 72 48 c2 f8 8d 6e c3 |..T...o. .rH...n.
28 43 3a c2 90 3f 6e c3  d0 e6 28 c2 c8 22 6f c3 |(C:..?n. ..(.."o.
d0 3d 11 c2 70 4a 72 c3  80 fa d4 c1 10 bc 7a c3 |.=..pJr. ......z.
00 f4 68 c0 20 c9 8a c3  00 99 68 43 c0 80 15 c4 |..h. ... ..hC....
00 78 eb c2 00 6c de c2  00 27 9a c2 a0 a3 22 c3 |.x...l.. .'....".
c8 c1 80 c2 30 ab 2f c3  f0 5c 66 c2 d8 a4 34 c3 |....0./. .\f...4.
b0 25 54 c2 28 9f 36 c3  50 4e 46 c2 30 2c 37 c3 |.%T.(.6. PNF.0,7.
10 f4 3a c2 cc f0 36 c3  48 2c 31 c2 94 3d 36 c3 |..:...6. H,1..=6.
90 76 28 c2 10 3e 35 c3  94 85 20 c2 fc 0b 34 c3 |.v(..>5. .. ...4.
fc 2a 19 c2 00 b8 32 c3  3c 45 12 c2 24 4c 31 c3 |.*....2. <E..$L1.
2c be 0b c2 e0 cf 2f c3  30 86 05 c2 38 48 2e c3 |,...../. 0...8H..
60 22 ff c1 ec b8 2c c3  e8 ac f3 c1 ba 24 2b c3 |`"....,. .....$+.
70 9e e8 c1 24 8d 29 c3  04 ed dd c1 f0 f3 27 c3 |p...$.). ......'.
58 89 d3 c1 58 5b 26 c3  e8 74 c9 c1 e1 c2 24 c3 |X...X[&. .t....$.
d0 a0 bf c1 57 2b 23 c3  ac 12 b6 c1 11 96 21 c3 |....W+#. ......!.
84 bf ac c1 d8 02 20 c3  2c a6 a3 c1 5f 72 1e c3 |...... . ,..._r..
8c c0 9a c1 8b e4 1c c3  24 0e 92 c1 45 5b 1b c3 |........ $...E[..
80 88 89 c1 0e d3 19 c3  94 34 81 c1 11 4f 18 c3 |........ .4...O..
c0 15 72 c1 65 ce 16 c3  90 16 62 c1 54 51 15 c3 |..r.e... ..b.TQ..
98 66 52 c1 da d7 13 c3  80 07 43 c1 f8 61 12 c3 |.fR..... ..C..a..
60 ef 33 c1 7e ef 10 c3  1c 20 25 c1 f6 80 0f c3 |`.3.~... . %.....
6c 9b 16 c1 ac 15 0e c3  d8 57 08 c1 0e ae 0c c3 |l....... .W......
48 a8 f4 c0 f7 49 0b c3  4c 1e d9 c0 60 e9 09 c3 |H....I.. L...`...
38 19 be c0 4b 8c 08 c3  9c 87 a3 c0 d4 32 07 c3 |8...K... .....2..
04 6d 89 c0 73 dc 05 c3  b8 8e 5f c0 dc 88 04 c3 |.m..s... .._.....
58 0e 2d c0 c8 39 03 c3  d0 d5 f6 bf 6d ed 01 c3 |X.-..9.. ....m...
20 43 95 bf 5d a4 00 c3  c0 4e d5 be f0 bc fe c2 | C..]... .N......
60 73 a4 3e 4c 37 fc c2  18 fe 85 3f b8 b7 f9 c2 |`s.>L7.. ...?....
60 6a e1 3f aa 3e f7 c2  cc ae 1d 40 30 cb f4 c2 |`j.?.>.. ...@0...
10 da 49 40 a2 5d f2 c2  c0 65 75 40 eb f5 ef c2 |..I@.].. .eu@....
3e 21 90 40 e8 93 ed c2  01 35 a5 40 8b 37 eb c2 |>!.@.... .5.@.7..
0b fa b9 40 74 e0 e8 c2  e1 63 ce 40 a7 8e e6 c2 |...@t... .c.@....
d7 7f e2 40 d8 42 e4 c2  7b 50 f6 40 e0 fb e1 c2 |...@.B.. {P.@....
63 ec 04 41 e7 b9 df c2  f0 85 0e 41 35 7d dd c2 |c..A.... ...A5}..
c7 fb 17 41 6d 45 db c2  fe 4f 21 41 10 12 d9 c2 |...AmE.. .O!A....
2a 79 2a 41 75 e4 d6 c2  0b 85 33 41 bc ba d4 c2 |*y*Au... ..3A....
a2 6b 3c 41 07 96 d2 c2  24 29 45 41 64 75 d0 c2 |.k<A.... $)EAdu..
a9 d0 4d 41 df 59 ce c2  83 53 56 41 4b 42 cc c2 |..MA.Y.. .SVAKB..
35 b5 5e 41 50 2f ca c2  cc f6 66 41 69 20 c8 c2 |5.^AP/.. ..fAi ..
df 1a 6f 41 c9 15 c6 c2  f5 1d 77 41 27 0f c4 c2 |..oA.... ..wA'...
2b 03 7f 41 ba 0c c2 c2  ef 68 83 41 2b 0e c0 c2 |+..A.... .h.A+...
bb 3c 87 41 2f 13 be c2  f4 03 8b 41 24 1c bc c2 |.<.A/... ...A$...
63 bd 8e 41 12 29 ba c2  d9 67 92 41 ad 39 b8 c2 |c..A.).. .g.A.9..
90 06 96 41 d0 4d b6 c2  46 96 99 41 1a 65 b4 c2 |...A.M.. F..A.e..
25 19 9d 41 6a 80 b2 c2  e1 8f a0 41 68 9f b0 c2 |%..Aj... ...Ah...
10 f9 a3 41 22 c1 ae c2  23 55 a7 41 00 e7 ac c2 |...A"... #U.A....
3e a5 aa 41 52 0f ab c2  0a ea ad 41 10 3b a9 c2 |>..AR... ...A.;..
67 20 b1 41 22 6a a7 c2  25 4c b4 41 f7 9b a5 c2 |g .A"j.. %L.A....
ea 6c b7 41 84 d1 a3 c2  22 83 ba 41 1a 0a a2 c2 |.l.A.... "..A....
ee 8a bd 41 95 44 a0 c2  80 87 c0 41 a3 82 9e c2 |...A.D.. ...A....
6e 7a c3 41 0c c3 9c c2  e6 62 c6 41 d6 06 9b c2 |nz.A.... .b.A....
44 3f c9 41 13 4d 99 c2  cc 11 cc 41 1f 96 97 c2 |D?.A.M.. ...A....
50 d8 ce 41 9e e1 95 c2  c0 98 d1 41 92 30 94 c2 |P..A.... ...A.0..
50 49 d4 41 32 81 92 c2  70 f2 d6 41 84 d4 90 c2 |PI.A2... p..A....
e0 92 d9 41 3c 2a 8f c2  c0 25 dc 41 a0 82 8d c2 |...A<*.. .%.A....
80 b2 de 41 60 dd 8b c2  40 35 e1 41 88 3a 8a c2 |...A`... @5.A.:..
00 ad e3 41 6c 99 88 c2  b0 1b e6 41 2c fb 86 c2 |...Al... ...A,...
c0 82 e8 41 18 5f 85 c2  d0 df ea 41 0c c5 83 c2 |...A._.. ...A....
f0 32 ed 41 40 2d 82 c2  10 7e ef 41 a3 97 80 c2 |.2.A@-.. .~.A....
58 c0 f1 41 ec 07 7e c2  98 f9 f3 41 ec e4 7a c2 |X..A..~. ...A..z.
70 2a f6 41 cc c5 77 c2  28 52 f8 41 36 ab 74 c2 |p*.A..w. (R.A6.t.
28 72 fa 41 2e 93 71 c2  b8 89 fc 41 b6 7f 6e c2 |(r.A..q. ...A..n.
c8 98 fe 41 02 70 6b c2  3c 50 00 42 fe 63 68 c2 |...A.pk. <P.B.ch.
d0 4f 01 42 d4 5b 65 c2  c6 4b 02 42 48 57 62 c2 |.O.B.[e. .K.BHWb.
2e 42 03 42 d4 54 5f c2  aa 36 04 42 cc 58 5c c2 |.B.B.T_. .6.B.X\.
1a 26 05 42 26 5d 59 c2  f2 11 06 42 f5 64 56 c2 |.&.B&]Y. ...B.dV.
8a fa 06 42 be 71 53 c2  c2 de 07 42 83 80 50 c2 |...B.qS. ...B..P.
f0 bf 08 42 e6 92 4d c2  7e 9f 09 42 36 aa 4a c2 |...B..M. ~..B6.J.
86 76 0a 42 c8 c0 47 c2  66 4c 0b 42 4f db 44 c2 |.v.B..G. fL.BO.D.
9e 1e 0c 42 19 fa 41 c2  40 ed 0c 42 7b 1b 3f c2 |...B..A. @..B{.?.
7e b8 0d 42 2f 3f 3c c2  56 80 0e 42 bb 65 39 c2 |~..B/?<. V..B.e9.
c4 44 0f 42 13 8f 36 c2  44 05 10 42 13 bb 33 c2 |.D.B..6. D..B..3.
d8 c2 10 42 9b e9 30 c2  8a 7d 11 42 4a 1b 2e c2 |...B..0. .}.BJ...
9a 34 12 42 88 4e 2b c2  e2 e7 12 42 87 84 28 c2 |.4.B.N+. ...B..(.
5c 98 13 42 22 bd 25 c2  8e 45 14 42 5e f7 22 c2 |\..B".%. .E.B^.".
72 ef 14 42 be 34 20 c2  8a 96 15 42 a2 74 1d c2 |r..B.4 . ...B.t..
b7 39 16 42 f0 b5 1a c2  0f da 16 42 68 f9 17 c2 |.9.B.... ...Bh...
08 77 17 42 9c 3f 15 c2  4d 11 18 42 62 87 12 c2 |.w.B.?.. M..Bb...
57 a8 18 42 9e d1 0f c2  ad 3c 19 42 60 1d 0d c2 |W..B.... .<.B`...
66 cd 19 42 98 6b 0a c2  ea 5b 1a 42 9a bb 07 c2 |f..B.k.. .[.B....
e0 e6 1a 42 bc 0c 05 c2  d3 6a 1b 42 cf 62 02 c2 |...B.... .j.B.b..
52 f4 1b 42 2b 6c ff c1  50 76 1c 42 1b 19 fa c1 |R..B+l.. Pv.B....
e6 f4 1c 42 27 cb f4 c1  ab 71 1d 42 55 7f ef c1 |...B'... .q.BU...
20 eb 1d 42 80 37 ea c1  12 62 1e 42 cf f1 e4 c1 | ..B.7.. .b.B....
6c d5 1e 42 89 b0 df c1  9a 46 1f 42 4c 71 da c1 |l..B.... .F.BLq..
b1 b4 1f 42 63 35 d5 c1  a0 1f 20 42 ef fa cf c1 |...Bc5.. .. B....
88 88 20 42 a6 c7 ca c1  46 ee 20 42 fa 93 c5 c1 |.. B.... F. B....
6c 51 21 42 50 62 c0 c1  3c b1 21 42 d4 32 bb c1 |lQ!BPb.. <.!B.2..
4c 0f 22 42 78 07 b6 c1  78 6b 22 42 54 df b0 c1 |L."Bx... xk"BT...
98 c2 22 42 62 b7 ab c1  20 18 23 42 34 92 a6 c1 |.."Bb...  .#B4...
10 6b 23 42 b0 6f a1 c1  68 bb 23 42 b4 4e 9c c1 |.k#B.o.. h.#B.N..
c8 08 24 42 94 30 97 c1  a0 53 24 42 54 12 92 c1 |..$B.0.. .S$BT...
00 9c 24 42 30 f9 8c c1  d0 e1 24 42 a0 e1 87 c1 |..$B0... ..$B....
a0 24 25 42 f0 c8 82 c1  20 65 25 42 10 66 7b c1 |.$%B....  e%B.f{.
00 a3 25 42 00 3f 71 c1  80 de 25 42 80 19 67 c1 |..%B.?q. ..%B..g.
00 18 26 42 80 f8 5c c1  00 4e 26 42 00 db 52 c1 |..&B..\. .N&B..R.
00 80 26 42 00 c1 48 c1  80 b1 26 42 c0 a1 3e c1 |..&B..H. ..&B..>.
c0 df 26 42 60 8e 34 c1  c0 0b 27 42 20 7c 2a c1 |..&B`.4. ..'B |*.
40 35 27 42 c0 66 20 c1  e0 5b 27 42 30 58 16 c1 |@5'B.f . .['B0X..
40 80 27 42 50 49 0c c1  a0 a4 27 42 48 3f 02 c1 |@.'BPI.. ..'BH?..
c0 c0 27 42 10 63 f0 c0  00 dd 27 42 d0 4f dc c0 |..'B.c.. ..'B.O..
20 f8 27 42 10 40 c8 c0  00 10 28 42 58 32 b4 c0 | .'B.@.. ..(BX2..
20 25 28 42 20 26 a0 c0  e0 37 28 42 78 1c 8c c0 | %(B &.. .7(Bx...
f0 47 28 42 00 2b 70 c0  b0 55 28 42 f8 22 48 c0 |.G(B.+p. .U(B."H.
40 61 28 42 d0 18 20 c0  10 69 28 42 70 12 f0 bf |@a(B.. . .i(Bp...
d0 6f 28 42 00 06 a0 bf  80 73 28 42 40 3b 20 bf |.o(B.... .s(B@; .
"""
fft_cmsis = convert_hexdump(hexdump_fft)
fft_cmsis = np.concatenate(
    (
        [fft_cmsis[0]],
        [fft_cmsis[i] + 1j*fft_cmsis[i + 1] for i in range(2, len(fft_cmsis), 2)],
        [fft_cmsis[1]]
    )
)
fft_cmsis_abs = np.abs(fft_cmsis)
fft_cmsis_angle = np.angle(fft_cmsis)

# compute RFFT using SciPy
fft_python = fft.rfft(input_sig, 1024)
fft_python_abs = np.abs(fft_python)
fft_python_angle = np.angle(fft_python)
# calculate max absolute and max relative error
amp_abs_err = calculate_max_abs_error(fft_python_abs, fft_cmsis_abs)
print(f"RFFT amplitude spectrum maximum absolute error: {amp_abs_err}")
amp_rel_err = calculate_max_rel_error(fft_python_abs, fft_cmsis_abs)
print(f"RFFT amplitude spectrum maximum relative error: {amp_rel_err}")
angle_abs_err = calculate_max_abs_error(fft_python_angle, fft_cmsis_angle)
print(f"RFFT phase spectrum maximum absolute error: {angle_abs_err}")
angle_rel_err = calculate_max_rel_error(fft_python_angle, fft_cmsis_angle)
print(f"RFFT phase spectrum maximum relative error: {angle_rel_err}")