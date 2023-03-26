# plot array obtained from hex dump
import numpy as np
import matplotlib.pyplot as plt


def plot_hexdump(hexdump):
    hexdump_str = " ".join(
        [line.split("|")[0].strip() for line in hexdump.split("\n") if line]
    )
    hexdump_bytes_str = "".join(
        [byte_str.strip() for byte_str in hexdump_str.split(" ")]
    )
    hexdump_bytes = bytearray.fromhex(hexdump_bytes_str)
    # convert hexdump bytes to float32 array
    arr = np.frombuffer(hexdump_bytes, dtype=np.float32)
    # plot the obtained array
    plt.figure()
    plt.plot(arr)
    plt.title("Hexdump array")
    plt.show()
    return arr


if __name__ == "__main__":
    hexdump = r"""
    a0 26 2f 4a b5 b4 ab 49  32 ff e2 48 98 71 e9 4a |.&/J...I 2..H.q.J
    93 84 8e 4b 9c 40 86 4b  24 87 c8 4a 86 d9 11 49 |...K.@.K $..J...I
    a8 24 81 46 12 13 e5 48  13 3f fc 49 74 02 e9 49 |.$.F...H .?.It..I
    20 da 9f 48 12 ba d4 47  dd 89 05 48 c9 d7 a2 48 | ..H...G ...H...H
    d2 5c 22 49 5b 94 43 48  c6 81 ea 46 da f9 06 48 |.\"I[.CH ...F...H
    13 10 7a 47 9c 5f 4c 48  90 68 de 47 e3 60 3b 46 |..zG._LH .h.G.`;F
    71 6a ef 47 dd 2a 0e 47  ee cd 15 47 75 ee 30 47 |qj.G.*.G ...Gu.0G
    f1 9f d0 45 20 4e b9 47  1e 9b 2f 47 7f 17 20 46 |...E N.G ../G.. F
    45 34 03 47 78 34 1f 44  58 a6 5b 47 c6 e1 1b 47 |E4.Gx4.D X.[G...G
    ca 84 2a 45 ac 1e 0e 47  29 60 b1 45 88 5d a1 46 |..*E...G )`.E.].F
    6b 82 f3 46 5b ad b3 44  12 6d 8e 46 5a d2 8c 45 |k..F[..D .m.FZ..E
    01 d5 1a 46 1f c8 91 46  98 2a 62 42 9e d1 a8 46 |...F...F .*bB...F
    f7 02 3a 46 e5 0f 1a 45  90 b6 4e 46 e5 1f 34 43 |..:F...E ..NF..4C
    a0 d8 55 46 e0 97 26 46  18 6e b5 44 90 31 3a 46 |..UF..&F .n.D.1:F
    25 53 6f 44 e7 0c fa 45  cc b3 f6 45 27 39 a8 44 |%SoD...E ...E'9.D
    70 cc 57 46 d8 66 31 45  9e c9 a9 45 90 aa 17 46 |p.WF.f1E ...E...F
    00 00 80 42 85 69 1b 46  c7 eb 74 45 82 b6 28 45 |...B.i.F ..tE..(E
    ff 2d f5 45 8e 25 31 42  da 3d 0b 46 95 7d 93 45 |.-.E.%1B .=.F.}.E
    18 70 0d 45 50 5a 2a 46  c4 e1 98 44 99 d3 86 45 |.p.EPZ*F ...D...E
    4a a6 86 45 3b 1b 8f 44  3c 89 0e 46 30 22 df 44 |J..E;..D <..F0".D
    54 5a 5e 45 8c 4b a6 45  78 52 92 43 b4 8f d8 45 |TZ^E.K.E xR.C...E
    8a 5f dd 44 ae d9 4e 45  ee b5 e0 45 46 1f a9 42 |._.D..NE ...EF..B
    e0 41 9b 45 35 66 19 45  ac 78 d1 44 59 73 b1 45 |.A.E5f.E .x.DYs.E
    5e a1 3f 43 84 e6 93 45  78 26 38 45 3c 23 9e 44 |^.?C...E x&8E<#.D
    ce bd c2 45 e8 68 ef 43  63 9c 89 45 bc 04 81 45 |...E.h.C c..E...E
    d4 5a 09 44 eb f4 c5 45  e0 ce 76 44 da d8 67 45 |.Z.D...E ..vD..gE
    29 73 b2 45 ca da 5d 42  fa a9 b4 45 21 a1 f7 44 |)s.E..]B ...E!..D
    10 d0 10 45 a6 50 c4 45  cc a8 00 43 45 f2 93 45 |...E.P.E ...CE..E
    51 c9 34 45 28 39 8c 44  a4 39 bc 45 d8 26 e8 43 |Q.4E(9.D .9.E.&.C
    ab 26 74 45 6d b2 7c 45  01 25 73 43 8f 8d 91 45 |.&tEm.|E .%sC...E
    f4 28 1b 44 0b e2 51 45  e6 4d 92 45 67 04 8b 42 |.(.D..QE .M.Eg..B
    fc 1b 92 45 18 be a7 44  c3 b4 0c 45 48 67 93 45 |...E...D ...EHg.E
    """
    arr = plot_hexdump(hexdump)