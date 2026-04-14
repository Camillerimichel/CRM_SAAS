from src.security.rbac import Access, pick_scope


def test_has_permission_accepts_descendant_scope():
    access = Access(
        user_type="staff",
        user_id=1,
        permissions={("data", "read")},
        societes={10},
        allowed_societes={10, 11, 12},
    )

    assert access.has_permission("data", "read", societe_id=10) is True
    assert access.has_permission("data", "read", societe_id=11) is True
    assert access.has_permission("data", "read", societe_id=12) is True
    assert access.has_permission("data", "read", societe_id=99) is False


def test_pick_scope_accepts_requested_descendant_scope():
    access = Access(
        user_type="staff",
        user_id=1,
        permissions={("data", "read")},
        societes={10},
        allowed_societes={10, 11, 12},
    )

    assert pick_scope(access, 11) == 11


def test_pick_scope_defaults_to_single_direct_scope():
    access = Access(
        user_type="staff",
        user_id=1,
        permissions={("data", "read")},
        societes={10},
        allowed_societes={10, 11, 12},
    )

    assert pick_scope(access, None) == 10


def test_global_scope_keeps_requested_scope():
    access = Access(
        user_type="staff",
        user_id=1,
        permissions={("data", "read")},
        societes={None},
        allowed_societes=set(),
    )

    assert pick_scope(access, None) is None
    assert pick_scope(access, 42) == 42
