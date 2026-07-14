//! State-free scalar expression IR shared by SQL binding, DML, streaming, and GPU lowering.

use gpu_db_sql::Decimal128;

/// A binary operator in the resident expression IR (arithmetic, comparison, or boolean). SQL binding,
/// DML predicate construction, streaming execution, and tests construct these nodes; lowering grows by
/// operator/type rather than matching whole query shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentBinaryOp {
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// SQL `LIKE`: the lhs is a text column, the rhs a [`ResidentExpr::TextLiteral`] pattern (`%` =
    /// any run, `_` = any one character). Byte-wise / UTF-8-char-aware (the type matrix, doc 19).
    Like,
}

/// The general scalar-expression IR the GPU interpreter evaluates. Column fields are table column
/// indices; the interpreter resolves device byte-offsets. Grows by node (literals of other types,
/// casts, unary ops, `LIKE`, ...), never by enumerating whole query shapes.
#[derive(Debug, Clone)]
pub(crate) enum ResidentExpr {
    Column(usize),
    Int4Literal(i32),
    /// A full-width i64 literal — the comparison value for an `int8`/`timestamp` column whose literal may
    /// exceed `i32` (a timestamp, a large bigint). Lowers to a `CompareScalarI64` VM step at `I64` element
    /// width (an `Int4Literal` against an int8 column widens instead, so this is only needed for literals
    /// outside the i32 range, but the DML predicate builder emits it for every int8 leaf for uniformity).
    Int8Literal(i64),
    /// A numeric (DECIMAL) literal as its [`Decimal128`] (mantissa + scale). Compared by rescaling to
    /// the target column's scale at lowering time (the type matrix, doc 19). An integer literal
    /// compared to a numeric column arrives as `Int4Literal` and is coerced to numeric on that path.
    NumericLiteral(Decimal128),
    /// A text (`text`/`varchar`) literal -- the comparison/LIKE value for a text column. Compared
    /// byte-wise (deterministic-collation equality is byte-identity; the type matrix, doc 19).
    TextLiteral(String),
    /// A boolean literal (`true`/`false`) -- the comparison value for `flag = true` / `flag = false`
    /// (the type matrix, doc 19). A bool column is a bitmap, so the comparison lowers to the
    /// bitmap->mask kernel with the appropriate `negate`.
    BoolLiteral(bool),
    /// `col IS NULL` / `col IS NOT NULL` (`is_not_null` selects which). A unary predicate LEAF over a
    /// column's NULL validity bitmap (M3 -- doc 21): it lowers to the SAME `gpu_db_resident_bool_to_mask`
    /// kernel as a bool column, pointed at the column's validity bitmap (1 = valid/present), with
    /// `negate = !is_not_null`. A column with no validity bitmap (no NULLs) lowers to an all-constant
    /// mask. Not an arithmetic operand -- the arith/compare paths reject it.
    IsNull {
        col: usize,
        is_not_null: bool,
    },
    Binary {
        op: ResidentBinaryOp,
        lhs: Box<ResidentExpr>,
        rhs: Box<ResidentExpr>,
    },
}
