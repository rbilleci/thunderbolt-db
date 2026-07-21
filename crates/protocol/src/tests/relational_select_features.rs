//! Relational SELECT predicate and DISTINCT parser coverage.

use super::*;

#[test]
fn parses_relational_select_in_membership_predicates_as_filter_groups() {
    assert_eq!(
        parse_command("SELECT id FROM people WHERE id IN (1, 3, 5) ORDER BY id").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }),
            filters: vec![SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Int4(1),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(1),
                }],
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(3),
                }],
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(5),
                }],
            ],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE name IN ('Ada', 'Grace') AND id >= 2")
            .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::Eq,
                value: SqlValue::Text("Ada".to_string()),
            }),
            filters: vec![
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Text("Ada".to_string()),
                },
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                },
            ],
            filter_groups: vec![
                vec![
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                ],
                vec![
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Eq,
                        value: SqlValue::Text("Grace".to_string()),
                    },
                    SelectFilter {
                        column: "id".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Int4(2),
                    },
                ],
            ],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT id FROM people WHERE id IN ()"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT id FROM people WHERE id NOT IN (1, 2)"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_select_between_predicates_as_filter_groups() {
    assert_eq!(
        parse_command("SELECT id FROM people WHERE id BETWEEN 2 AND 4 ORDER BY id").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "id".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Int4(2),
            }),
            filters: vec![
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                },
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Lte,
                    value: SqlValue::Int4(4),
                },
            ],
            filter_groups: vec![vec![
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Int4(2),
                },
                SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Lte,
                    value: SqlValue::Int4(4),
                },
            ]],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE name BETWEEN 'Ada' AND 'Grace' OR id = 4")
            .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::Gte,
                value: SqlValue::Text("Ada".to_string()),
            }),
            filters: vec![
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Gte,
                    value: SqlValue::Text("Ada".to_string()),
                },
                SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::Lte,
                    value: SqlValue::Text("Grace".to_string()),
                },
            ],
            filter_groups: vec![
                vec![
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Gte,
                        value: SqlValue::Text("Ada".to_string()),
                    },
                    SelectFilter {
                        column: "name".to_string(),
                        op: SelectFilterOp::Lte,
                        value: SqlValue::Text("Grace".to_string()),
                    },
                ],
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(4),
                }],
            ],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT id FROM people WHERE id NOT BETWEEN 1 AND 3"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_select_prefix_like_predicates_as_filters() {
    assert_eq!(
        parse_command("SELECT id FROM people WHERE name LIKE 'Gra%' ORDER BY id").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("Gra".to_string()),
            }),
            filters: vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("Gra".to_string()),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("Gra".to_string()),
            }]],
            order_by: vec![SelectOrder {
                column: "id".to_string(),
                descending: false,
            }],
            limit: None,
            offset: None,
        })
    );

    assert_eq!(
        parse_command("SELECT name FROM people WHERE name LIKE 'A%' OR id = 3").unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: false,
            projection: SelectProjection::Columns(vec!["name".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("A".to_string()),
            }),
            filters: vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("A".to_string()),
            }],
            filter_groups: vec![
                vec![SelectFilter {
                    column: "name".to_string(),
                    op: SelectFilterOp::LikePrefix,
                    value: SqlValue::Text("A".to_string()),
                }],
                vec![SelectFilter {
                    column: "id".to_string(),
                    op: SelectFilterOp::Eq,
                    value: SqlValue::Int4(3),
                }],
            ],
            order_by: Vec::new(),
            limit: None,
            offset: None,
        })
    );

    assert!(matches!(
        parse_command("SELECT id FROM people WHERE name NOT LIKE 'A%'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT id FROM people WHERE name LIKE '%da'"),
        Err(ParseError::InvalidRelationalSql)
    ));
    assert!(matches!(
        parse_command("SELECT id FROM people WHERE name LIKE 'A_a%'"),
        Err(ParseError::InvalidRelationalSql)
    ));
}

#[test]
fn parses_relational_select_distinct_projection() {
    assert_eq!(
        parse_command(
            "SELECT DISTINCT name, id FROM people WHERE name LIKE 'G%' ORDER BY name DESC LIMIT 2 OFFSET 1",
        )
        .unwrap(),
        Command::Select(Select {
            table: "people".to_string(),
            public_only: false,
            distinct: true,
            projection: SelectProjection::Columns(vec!["name".to_string(), "id".to_string()]),
            group_by: None,
            having_groups: Vec::new(),
            filter: Some(SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("G".to_string()),
            }),
            filters: vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("G".to_string()),
            }],
            filter_groups: vec![vec![SelectFilter {
                column: "name".to_string(),
                op: SelectFilterOp::LikePrefix,
                value: SqlValue::Text("G".to_string()),
            }]],
            order_by: vec![SelectOrder {
                column: "name".to_string(),
                descending: true,
            }],
            limit: Some(2),
            offset: Some(1),
        })
    );

    assert!(matches!(
        parse_command("SELECT DISTINCT * FROM people"),
        Err(ParseError::InvalidRelationalSql)
    ));
}
