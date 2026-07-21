//! Relational aggregate parser coverage.

use super::*;

#[test]
fn parses_relational_count_aggregates() {
    assert_eq!(
        parse_command(
            "SELECT name, COUNT(*) FROM people WHERE id >= 2 GROUP BY name HAVING count >= 2 OR name = 'Ada' ORDER BY count DESC LIMIT 2 OFFSET 1",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::GroupedCount {
                column: "name".to_string(),
            },
            group_by: Some("name".to_string()),
            having_groups: vec![
                vec![SelectFilter {
                    column: "count".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                }],
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                }],
            ],
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "count".to_string(),
                descending: true,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert_eq!(
        parse_command("SELECT COUNT(*) FROM people").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::CountAll,
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT COUNT(*), id FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_having_aggregates() {
    assert_eq!(
        parse_command(
            "SELECT name, SUM(id) FROM people GROUP BY name HAVING sum > 2 AND name LIKE 'G%' ORDER BY sum DESC",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::GroupedSum {
                group_column: "name".to_string(),
                sum_column: "id".to_string(),
            },
            group_by: Some("name".to_string()),
            having_groups: vec![vec![
                SelectFilter {
                    column: "sum".to_string(),
                    op: SelectFilterOp::Gt,
                    value: SqlValue::Int4(2),
                },
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("G".to_string()),
                },
            ]],
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: vec![SelectOrder {
                column: "sum".to_string(),
                descending: true,
            }],
            limit: None,
            offset: None,
        })
    );
}

#[test]
fn parses_relational_sum_aggregates() {
    assert_eq!(
        parse_command(
            "SELECT name, SUM(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY sum DESC LIMIT 2 OFFSET 1",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::GroupedSum {
                group_column: "name".to_string(),
                sum_column: "id".to_string(),
            },
            group_by: Some("name".to_string()),
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "sum".to_string(),
                descending: true,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert_eq!(
        parse_command("SELECT SUM(id) FROM people").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Sum {
                column: "id".to_string(),
            },
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT SUM(id), name FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_avg_aggregates() {
    assert_eq!(
        parse_command(
            "SELECT name, AVG(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY avg DESC LIMIT 2 OFFSET 1",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::GroupedAvg {
                group_column: "name".to_string(),
                avg_column: "id".to_string(),
            },
            group_by: Some("name".to_string()),
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "avg".to_string(),
                descending: true,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert_eq!(
        parse_command("SELECT AVG(id) FROM people").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Avg {
                column: "id".to_string(),
            },
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT AVG(id), name FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT DISTINCT AVG(id) FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_min_max_aggregates() {
    assert_eq!(
        parse_command(
            "SELECT name, MIN(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY min DESC LIMIT 2 OFFSET 1",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::GroupedMin {
                group_column: "name".to_string(),
                min_column: "id".to_string(),
            },
            group_by: Some("name".to_string()),
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }]],
            order_by: vec![SelectOrder {
                column: "min".to_string(),
                descending: true,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert_eq!(
        parse_command("SELECT MAX(name) FROM people").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Max {
                column: "name".to_string(),
            },
            group_by: None,
            having_groups: Vec::new(),
            filter: None,
            filters: Vec::new(),
            filter_groups: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT MIN(id), name FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}
