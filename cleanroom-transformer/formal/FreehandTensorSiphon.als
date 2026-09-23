module FreehandTensorSiphon

// A finite relational model of the structural tensor siphon induced by
// matrix multiplication.  Alloy searches for counterexamples to the
// stated shape/contraction invariants.

sig Dim {}

sig Matrix {
  rows: one Dim,
  cols: one Dim
}

sig Product {
  left: one Matrix,
  right: one Matrix,
  resultRows: one Dim,
  resultCols: one Dim,
  siphoned: one Dim
}

// Matrix multiplication is defined only when the inner dimensions agree.
pred Compatible[p: Product] {
  p.left.cols = p.right.rows
}

// The result preserves the outer dimensions and consumes the shared one.
pred MatmulSiphon[p: Product] {
  Compatible[p]
  p.resultRows = p.left.rows
  p.resultCols = p.right.cols
  p.siphoned = p.left.cols
}

// Canonical freehand tensor interpretation.
sig FreehandTensor {
  freeLeft: one Dim,
  contracted: one Dim,
  freeRight: one Dim
}

sig Representation {
  product: one Product,
  tensor: one FreehandTensor
}

pred Represents[r: Representation] {
  MatmulSiphon[r.product]
  r.tensor.freeLeft = r.product.left.rows
  r.tensor.contracted = r.product.siphoned
  r.tensor.freeRight = r.product.right.cols
}

assert SiphonPreservesOuterDimensions {
  all p: Product |
    MatmulSiphon[p] implies
      p.resultRows = p.left.rows and
      p.resultCols = p.right.cols
}

assert SiphonConsumesSharedDimension {
  all p: Product |
    MatmulSiphon[p] implies
      p.siphoned = p.left.cols and
      p.siphoned = p.right.rows
}

assert RepresentationIsShapeFaithful {
  all r: Representation |
    Represents[r] implies
      r.tensor.freeLeft = r.product.resultRows and
      r.tensor.freeRight = r.product.resultCols
}

// Counterlemma schema: assumptions hold, but a claimed conclusion fails.
pred CounterlemmaOuterDimension[p: Product] {
  MatmulSiphon[p]
  some p.resultRows
  p.resultRows != p.left.rows
}

pred CounterlemmaContractedDimension[p: Product] {
  MatmulSiphon[p]
  p.siphoned != p.left.cols
}

check SiphonPreservesOuterDimensions for 5
check SiphonConsumesSharedDimension for 5
check RepresentationIsShapeFaithful for 5

// These should be UNSAT if the semantics above are coherent.
run CounterlemmaOuterDimension for 5
run CounterlemmaContractedDimension for 5
