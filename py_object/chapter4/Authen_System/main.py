import module

if __name__ == '__main__':
    # 模块经过登录，授权方可通过
    module.authenticator.add_user("joe", "joepassword")
    module.authorizor.add_pemission("paint")

    # not logged in
    # module.authorizor.check_permission("paint", "joe")

    print(module.authenticator.is_logged_in("joe"))
    # password wrong
    # module.authenticator.log_in("joe", "joespassword")

    module.authenticator.log_in("joe", "joepassword")

    # not permitted
    # module.authorizor.check_permission("paint", "joe")

    module.authorizor.permit_user("paint", "joe")
    print(module.authorizor.check_permission("paint", "joe"))