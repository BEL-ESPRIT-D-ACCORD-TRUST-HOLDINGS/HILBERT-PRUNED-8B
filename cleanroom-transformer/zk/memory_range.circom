pragma circom 2.1.6;

// Zero-knowledge proof that a contiguous run of decision-memory entries exists (SPEC.md 6.5).
//
// Public:  root            Poseidon Merkle root over the memory's entry leaves (position = seq)
//          startIndex      first position of the run
//          rangeSize       1 <= rangeSize <= MAX_RANGE_SIZE
// Private: leaves[i]       entry leaves at startIndex + i (slots i >= rangeSize are ignored)
//          siblings[i][d]  Merkle path of each leaf
// Output:  rangeCommitment Poseidon chain over the active leaves, binding the proof to exactly
//                          those entries without revealing them.
//
// Each active slot carries its own Merkle path, and the path directions are the bits of its
// position, so the leaves are proven to sit exactly at startIndex .. startIndex + rangeSize - 1.
// Empty positions hold 0; active leaves must be nonzero, so a run cannot extend past the log.
// (An earlier sketch hashed the run into one value and walked a single path with free direction
// bits; that value is not a node of the tree, and the directions were not tied to startIndex.)

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/comparators.circom";

function bitsFor(x) {
    var n = 1;
    while ((1 << n) <= x) n++;
    return n;
}

// Root of the Merkle path from `leaf` at `index` (bit d of index: 0 = current node is the left child).
template MerklePath(DEPTH) {
    signal input leaf;
    signal input index;
    signal input siblings[DEPTH];
    signal output root;

    component bits = Num2Bits(DEPTH); // also enforces index < 2^DEPTH
    bits.in <== index;

    signal cur[DEPTH + 1];
    signal left[DEPTH];
    signal right[DEPTH];
    component h[DEPTH];
    cur[0] <== leaf;
    for (var d = 0; d < DEPTH; d++) {
        left[d] <== cur[d] + bits.out[d] * (siblings[d] - cur[d]);
        right[d] <== siblings[d] + bits.out[d] * (cur[d] - siblings[d]);
        h[d] = Poseidon(2);
        h[d].inputs[0] <== left[d];
        h[d].inputs[1] <== right[d];
        cur[d + 1] <== h[d].out;
    }
    root <== cur[DEPTH];
}

template MemoryRangeProof(DEPTH, MAX_RANGE_SIZE) {
    assert(MAX_RANGE_SIZE >= 1 && MAX_RANGE_SIZE <= (1 << DEPTH));
    var NB = bitsFor(MAX_RANGE_SIZE);

    signal input root;
    signal input startIndex;
    signal input rangeSize;
    signal input leaves[MAX_RANGE_SIZE];
    signal input siblings[MAX_RANGE_SIZE][DEPTH];
    signal output rangeCommitment;

    // Bit-decompose first so the comparators below only ever see small numbers
    // (circomlib comparators are unsound for inputs wider than their bit width).
    component startBits = Num2Bits(DEPTH);
    startBits.in <== startIndex;
    component sizeBits = Num2Bits(NB);
    sizeBits.in <== rangeSize;

    component atLeastOne = GreaterEqThan(NB);
    atLeastOne.in[0] <== rangeSize;
    atLeastOne.in[1] <== 1;
    atLeastOne.out === 1;

    component atMostMax = LessEqThan(NB);
    atMostMax.in[0] <== rangeSize;
    atMostMax.in[1] <== MAX_RANGE_SIZE;
    atMostMax.out === 1;

    component fits = LessEqThan(DEPTH + 1); // startIndex + rangeSize <= 2^DEPTH
    fits.in[0] <== startIndex + rangeSize;
    fits.in[1] <== 1 << DEPTH;
    fits.out === 1;

    component active[MAX_RANGE_SIZE];
    component path[MAX_RANGE_SIZE];
    component isEmpty[MAX_RANGE_SIZE];
    component step[MAX_RANGE_SIZE];
    signal index[MAX_RANGE_SIZE];
    signal accum[MAX_RANGE_SIZE + 1];
    accum[0] <== 0;

    for (var i = 0; i < MAX_RANGE_SIZE; i++) {
        active[i] = LessThan(NB); // slot i is inside the run
        active[i].in[0] <== i;
        active[i].in[1] <== rangeSize;

        index[i] <== active[i].out * (startIndex + i); // inactive slots use position 0
        path[i] = MerklePath(DEPTH);
        path[i].leaf <== leaves[i];
        path[i].index <== index[i];
        for (var d = 0; d < DEPTH; d++) path[i].siblings[d] <== siblings[i][d];
        active[i].out * (path[i].root - root) === 0;

        isEmpty[i] = IsZero(); // a recorded entry is never the empty leaf 0
        isEmpty[i].in <== leaves[i];
        active[i].out * isEmpty[i].out === 0;

        // masked accumulator: accum advances only over active slots
        step[i] = Poseidon(2);
        step[i].inputs[0] <== accum[i];
        step[i].inputs[1] <== leaves[i];
        accum[i + 1] <== accum[i] + active[i].out * (step[i].out - accum[i]);
    }
    rangeCommitment <== accum[MAX_RANGE_SIZE];
}

// 2^16 = 65,536 memory entries, runs of up to 8.
component main {public [root, startIndex, rangeSize]} = MemoryRangeProof(16, 8);
