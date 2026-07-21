use super::*;

#[test]
fn projection_binding_distinguishes_quoted_star_from_mixed_wildcard() {
    let table = catalog_relation_table(
        "public",
        "star_projection_contract",
        &[("*", SqlType::Int4), ("id", SqlType::Int4)],
    );
    let bind = |sql: &str| {
        let Command::Select(select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        bind_relational_select(&table, &select).unwrap()
    };

    let quoted = bind(r#"SELECT "*" FROM star_projection_contract"#);
    assert_eq!(quoted.selected_indexes, vec![0]);
    assert_eq!(quoted.selected_columns[0].name, "*");

    let mixed = bind("SELECT id, * FROM star_projection_contract");
    assert_eq!(mixed.selected_indexes, vec![1, 0, 1]);
    assert_eq!(
        mixed
            .selected_columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["id", "*", "id"]
    );
}
