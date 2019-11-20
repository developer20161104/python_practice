import hashlib
import excepts


# 用户对象
class User:
    def __init__(self, username, password):
        self.username = username
        self.password = self._encrypt_pw(password)
        self.is_logged_in = False

    def _encrypt_pw(self, password):
        hash_string = (self.username + password).encode("utf8")

        return hashlib.sha256(hash_string).hexdigest()

    def check_password(self, password):
        encrypt = self._encrypt_pw(password)
        return encrypt == self.password


# 关联用户名与用户对象
class Authenticator:
    def __init__(self):
        self.users = {}

    def add_user(self, username, password):
        if username in self.users:
            raise excepts.UsernameAlreadyExists(username)
        if len(password) < 6:
            raise excepts.PasswordTooShort(username)
        self.users[username] = User(username, password)

    # 第一步，登录
    def log_in(self, username, password):
        try:
            user = self.users[username]
        except KeyError:
            raise excepts.InvalidUsername(username)

        if not user.check_password(password):
            raise excepts.InvalidPassword(username)

        user.is_logged_in = True
        return True

    def is_logged_in(self, username):
        if username in self.users:
            return self.users[username].is_logged_in
        return False


# 保存权限与用户之间的关系
class Authorizor:
    def __init__(self, authenticators):
        self.authenticator = authenticators
        self.permissions = {}

    # 给用户添加权限
    # 第二步，授权
    def add_pemission(self, perm_name):
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            #  防止
            self.permissions[perm_name] = set()
        else:
            raise PermissionError("Permission Exist")

    # 为权限指定用户
    def permit_user(self, perm_name, username):
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist")
        else:
            if username not in self.authenticator.users:
                raise excepts.InvalidUsername(username)
            perm_set.add(username)

    # 检查用户特定权限
    def check_permission(self, perm_name, username):
        if not self.authenticator.is_logged_in(username):
            raise excepts.NotLoggedInError(username)

        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("permission not exist")
        else:
            if username not in perm_set:
                raise excepts.NotPermittedError(username)
            else:
                return True


authenticator = Authenticator()
authorizor = Authorizor(authenticator)
