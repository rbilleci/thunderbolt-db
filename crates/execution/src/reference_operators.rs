//! Host iterator operators retained only as the interim `cfg(test)` MVCC semantic oracle.
//! Their eventual deletion is owned by PLAN task RETIRE-001; they are not a product execution path.

pub trait Operator<Row = Vec<u8>> {
    fn open(&mut self) {}
    fn next(&mut self) -> Option<Row>;
    fn close(&mut self) {}
}

#[derive(Debug, Default)]
pub struct CpuNoop;

impl Operator<Vec<u8>> for CpuNoop {
    fn next(&mut self) -> Option<Vec<u8>> {
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct VecOperator {
    rows: Vec<Vec<u8>>,
    next_index: usize,
}

impl VecOperator {
    pub fn new(rows: Vec<Vec<u8>>) -> Self {
        Self {
            rows,
            next_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl Operator<Vec<u8>> for VecOperator {
    fn open(&mut self) {
        self.next_index = 0;
    }

    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.rows.len();
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScanOperator<Row> {
    rows: Vec<Row>,
    next_index: usize,
}

impl<Row> ScanOperator<Row> {
    pub fn new(rows: Vec<Row>) -> Self {
        Self {
            rows,
            next_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl<Row: Clone> Operator<Row> for ScanOperator<Row> {
    fn open(&mut self) {
        self.next_index = 0;
    }

    fn next(&mut self) -> Option<Row> {
        let row = self.rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.rows.len();
    }
}

pub struct FilterOperator<Row, Child, Predicate> {
    child: Child,
    predicate: Predicate,
    _row: std::marker::PhantomData<Row>,
}

impl<Row, Child, Predicate> FilterOperator<Row, Child, Predicate> {
    pub fn new(child: Child, predicate: Predicate) -> Self {
        Self {
            child,
            predicate,
            _row: std::marker::PhantomData,
        }
    }
}

impl<Row, Child, Predicate> Operator<Row> for FilterOperator<Row, Child, Predicate>
where
    Child: Operator<Row>,
    Predicate: FnMut(&Row) -> bool,
{
    fn open(&mut self) {
        self.child.open();
    }

    fn next(&mut self) -> Option<Row> {
        while let Some(row) = self.child.next() {
            if (self.predicate)(&row) {
                return Some(row);
            }
        }
        None
    }

    fn close(&mut self) {
        self.child.close();
    }
}

pub struct ProjectOperator<Input, Output, Child, Projection> {
    child: Child,
    projection: Projection,
    _input: std::marker::PhantomData<Input>,
    _output: std::marker::PhantomData<Output>,
}

impl<Input, Output, Child, Projection> ProjectOperator<Input, Output, Child, Projection> {
    pub fn new(child: Child, projection: Projection) -> Self {
        Self {
            child,
            projection,
            _input: std::marker::PhantomData,
            _output: std::marker::PhantomData,
        }
    }
}

impl<Input, Output, Child, Projection> Operator<Output>
    for ProjectOperator<Input, Output, Child, Projection>
where
    Child: Operator<Input>,
    Projection: FnMut(Input) -> Output,
{
    fn open(&mut self) {
        self.child.open();
    }

    fn next(&mut self) -> Option<Output> {
        self.child.next().map(&mut self.projection)
    }

    fn close(&mut self) {
        self.child.close();
    }
}

pub struct LimitOperator<Row, Child> {
    child: Child,
    remaining: usize,
    initial_limit: usize,
    _row: std::marker::PhantomData<Row>,
}

impl<Row, Child> LimitOperator<Row, Child> {
    pub fn new(child: Child, limit: usize) -> Self {
        Self {
            child,
            remaining: limit,
            initial_limit: limit,
            _row: std::marker::PhantomData,
        }
    }
}

impl<Row, Child> Operator<Row> for LimitOperator<Row, Child>
where
    Child: Operator<Row>,
{
    fn open(&mut self) {
        self.remaining = self.initial_limit;
        self.child.open();
    }

    fn next(&mut self) -> Option<Row> {
        if self.remaining == 0 {
            return None;
        }

        let row = self.child.next()?;
        self.remaining -= 1;
        Some(row)
    }

    fn close(&mut self) {
        self.remaining = 0;
        self.child.close();
    }
}

pub struct SortOperator<Row, Child, Compare> {
    child: Child,
    compare: Compare,
    sorted_rows: Vec<Row>,
    next_index: usize,
}

impl<Row, Child, Compare> SortOperator<Row, Child, Compare> {
    pub fn new(child: Child, compare: Compare) -> Self {
        Self {
            child,
            compare,
            sorted_rows: Vec::new(),
            next_index: 0,
        }
    }
}

impl<Row, Child, Compare> Operator<Row> for SortOperator<Row, Child, Compare>
where
    Row: Clone,
    Child: Operator<Row>,
    Compare: FnMut(&Row, &Row) -> std::cmp::Ordering,
{
    fn open(&mut self) {
        self.child.open();
        self.sorted_rows.clear();
        self.next_index = 0;

        while let Some(row) = self.child.next() {
            self.sorted_rows.push(row);
        }

        self.sorted_rows.sort_by(&mut self.compare);
        self.child.close();
    }

    fn next(&mut self) -> Option<Row> {
        let row = self.sorted_rows.get(self.next_index)?.clone();
        self.next_index += 1;
        Some(row)
    }

    fn close(&mut self) {
        self.next_index = self.sorted_rows.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_operator_yields_rows_in_order() {
        let mut op = VecOperator::new(vec![b"row-1".to_vec(), b"row-2".to_vec()]);

        assert_eq!(op.len(), 2);
        assert!(!op.is_empty());
        assert_eq!(op.next(), Some(b"row-1".to_vec()));
        assert_eq!(op.next(), Some(b"row-2".to_vec()));
        assert_eq!(op.next(), None);
    }

    #[test]
    fn vec_operator_open_rewinds_and_close_exhausts() {
        let mut op = VecOperator::new(vec![b"row-1".to_vec()]);

        assert_eq!(op.next(), Some(b"row-1".to_vec()));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(b"row-1".to_vec()));

        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn scan_operator_yields_rows_in_order() {
        let mut op = ScanOperator::new(vec![1_u32, 2_u32, 3_u32]);

        assert_eq!(op.len(), 3);
        assert!(!op.is_empty());
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(3));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(1));
        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn filter_operator_skips_non_matching_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let mut op = FilterOperator::new(scan, |row: &i32| row % 2 == 0);

        op.open();
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(4));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn project_operator_maps_child_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32]);
        let mut op = ProjectOperator::new(scan, |row| format!("row-{row}"));

        op.open();
        assert_eq!(op.next(), Some("row-1".to_string()));
        assert_eq!(op.next(), Some("row-2".to_string()));
        assert_eq!(op.next(), Some("row-3".to_string()));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn limit_operator_caps_child_rows() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let mut op = LimitOperator::new(scan, 2);

        op.open();
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), None);

        op.open();
        assert_eq!(op.next(), Some(1));
        op.close();
        assert_eq!(op.next(), None);
    }

    #[test]
    fn scan_filter_project_pipeline_composes() {
        let scan = ScanOperator::new(vec![1_i32, 2_i32, 3_i32, 4_i32]);
        let filter = FilterOperator::new(scan, |row: &i32| row % 2 == 1);
        let limit = LimitOperator::new(filter, 1);
        let mut op = ProjectOperator::new(limit, |row| row * 10);

        op.open();
        assert_eq!(op.next(), Some(10));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn sort_operator_orders_child_rows() {
        let scan = ScanOperator::new(vec![3_i32, 1_i32, 4_i32, 2_i32]);
        let mut op = SortOperator::new(scan, |left: &i32, right: &i32| left.cmp(right));

        op.open();
        assert_eq!(op.next(), Some(1));
        assert_eq!(op.next(), Some(2));
        assert_eq!(op.next(), Some(3));
        assert_eq!(op.next(), Some(4));
        assert_eq!(op.next(), None);
        op.close();
    }

    #[test]
    fn scan_filter_sort_limit_project_pipeline_composes() {
        let scan = ScanOperator::new(vec![4_i32, 1_i32, 3_i32, 2_i32]);
        let filter = FilterOperator::new(scan, |row: &i32| row % 2 == 0);
        let sort = SortOperator::new(filter, |left: &i32, right: &i32| right.cmp(left));
        let limit = LimitOperator::new(sort, 1);
        let mut op = ProjectOperator::new(limit, |row| row * 10);

        op.open();
        assert_eq!(op.next(), Some(40));
        assert_eq!(op.next(), None);
        op.close();
    }
}
